from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import os
import logging
import numpy as np
import psutil
import torch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename
import time
import soundfile as sf
import io
import re
import nltk
from spellchecker import SpellChecker
from num2words import num2words
from dateutil import parser
from collections import deque
from threading import Lock

# Initialize NLTK and download necessary data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Import TTS API
from TTS.api import TTS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
GEN_TEMP = 0.6
SAMPLE_RATE = 24000
PROCESSING_DELAY = 0.5
MEMORY_CHECK_INTERVAL = 60
SPEAKERS_FOLDER = "speakers"
UPLOAD_FOLDER = "uploads"
TEMPLATE_FOLDER = "templates"
MIN_SENTENCE_LENGTH = 1  # Changed to 1 to allow shorter inputs

# Create Flask app and SocketIO instance
app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app, async_mode='eventlet')

# Load TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
logger.info(f"TTS model loaded on device: {device}")

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=4)

# Initialize lemmatizer and spell checker
lemmatizer = nltk.WordNetLemmatizer()
spell = SpellChecker()

# Global variables
text_buffer = ""
paused = False
sentence_queue = deque()
queue_lock = Lock()
correction_flag = False
last_processed_text = ""

@lru_cache(maxsize=128)
def get_speaker_embedding(speaker_wav_path):
    wav, _ = sf.read(speaker_wav_path)
    return model.synthesizer.tts_model.speaker_manager.compute_embedding(wav)

def split_into_sentences(text):
    sentences = re.split(r'[।.!?]\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def remove_special_characters(text):
    return re.sub(r'[./"\'=\-*!^%$#?]', '', text)

def normalize_text(text):
    return text  # For non-Latin scripts, return the text as-is

def correct_spelling(text):
    if not text:
        return ""
    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return ' '.join(corrected_words)

def process_dates(text):
    if not text:
        return ""
    date_pattern = r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b'
    
    def date_repl(match):
        date_str = match.group(1)
        try:
            date = parser.parse(date_str, dayfirst=True, yearfirst=False)
            day_num = date.day
            day_str = f"{day_num}{'th' if 11 <= day_num <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day_num % 10, 'th')}"
            month_str = date.strftime("%B")
            year = date.year
            # Adjust for two-digit years
            if year < 100:
                year += 2000 if year <= 24 else 1900  # Adjust this threshold as needed
            year_str = num2words(year)
            return f"the {day_str} of {month_str} {year_str}"
        except ValueError:
            return match.group()  # Return the original string if parsing fails
    
    return re.sub(date_pattern, date_repl, text)

def normalize_text(text):
    if not text:
        return ""
    words = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in words])

def preprocess_text(text):
    if text is None:
        return ""
    
    try:
        # Existing preprocessing steps
        text = re.sub(r'<original_message>.*?</original_message>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'<function_calls>.*?</function_calls>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<function_name>.*?</function_name>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\b(end_call|start_call|function_name)\b', '', text, flags=re.IGNORECASE)
        
        # New step: Remove {prompt} and {response} placeholders
        text = re.sub(r'\{prompt\}|\{response\}', '', text)
        
        text = process_dates(text)
        text = remove_special_characters(text)
        text = normalize_text(text)
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return ""
    
def process_text(text, speaker_wav, language):
    try:
        logger.info(f"Starting process_text with text: '{text}', language: {language}")
        if not text:
            logger.warning("Empty text input, returning empty audio")
            return np.array([]), 0, 0

        start_time = time.time()
        processed_text = preprocess_text(text)
        logger.info(f"Processed text: '{processed_text}'")
        
        if not processed_text:
            logger.warning(f"Processed text is empty for input: '{text}'")
            return np.array([]), 0, 0
        
        logger.info("Calling model.tts")
        audio = model.tts(text=processed_text, speaker_wav=speaker_wav, language=language)
        logger.info(f"model.tts returned. Audio length: {len(audio)}")
        audio_array = np.array(audio)
        
        processing_time = time.time() - start_time
        audio_duration = len(audio_array) / SAMPLE_RATE
        real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0

        logger.info(f"Original text: '{text}'")
        logger.info(f"Processed text: '{processed_text}'")
        logger.info(f"Processing time: {processing_time:.3f} seconds")
        logger.info(f"Real-time factor: {real_time_factor:.3f}")

        return audio_array, processing_time, real_time_factor
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        logger.error(f"Text causing error: '{text}'")
        return np.array([]), 0, 0

def check_memory_usage():
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 90:
        logger.warning(f"High memory usage: {memory_usage}%")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speakers/')
def get_speakers():
    speaker_files = os.listdir(SPEAKERS_FOLDER)
    return jsonify(speaker_files)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/process_audio/', methods=['POST'])
def process_audio():
    file = request.files['file']
    language = request.form.get('language', 'en')
    text = request.form.get('text', '')

    if file:
        filename = secure_filename(file.filename)
        speaker_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(speaker_path)
        
        audio_array, processing_time, real_time_factor = process_text(text, speaker_path, language)
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.wav')
        sf.write(output_path, audio_array, SAMPLE_RATE)
        
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'output.wav', as_attachment=True)
    
    return 'File not found', 400

@app.route('/play_audio/', methods=['POST'])
def play_audio():
    global paused
    paused = False
    emit('play', broadcast=True)
    return 'Playing audio'

@app.route('/pause_audio/', methods=['POST'])
def pause_audio():
    global paused
    paused = True
    emit('pause', broadcast=True)
    return 'Pausing audio'

@app.route('/resume_audio/', methods=['POST'])
def resume_audio():
    global paused
    paused = False
    emit('resume', broadcast=True)
    return 'Resuming audio'

@app.route('/stop_audio/', methods=['POST'])
def stop_audio():
    global paused
    paused = True
    emit('stop', broadcast=True)
    return 'Stopping audio'

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    emit('status', {'message': 'Disconnected'})

@socketio.on('message')
def handle_message(message):
    global text_buffer, paused
    if not paused:
        text_buffer += message
        emit('message', {'message': message}, broadcast=True)

@socketio.on('correction')
def handle_correction(data):
    global correction_flag
    correction_flag = True
    corrected_text = data['corrected_text']
    original_text = data['original_text']
    emit('correction', {'corrected_text': corrected_text, 'original_text': original_text}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
