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
    speakers = [os.path.splitext(f)[0] for f in os.listdir(SPEAKERS_FOLDER) if f.endswith('.wav')]
    return jsonify(speakers)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'speakerFile' not in request.files:
        return 'No file part', 400

    file = request.files['speakerFile']
    if file.filename == '':
        return 'No selected file', 400

    language = request.form.get('language')
    if not language:
        return 'No language specified', 400

    os.makedirs(SPEAKERS_FOLDER, exist_ok=True)
    file_path = os.path.join(SPEAKERS_FOLDER, secure_filename(file.filename))
    file.save(file_path)
    return jsonify({'message': 'File uploaded successfully', 'speaker_file': file.filename}), 200

@socketio.on('connect')
def handle_connect():
    emit('response', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('message')
def handle_message(data):
    global text_buffer, paused, sentence_queue, correction_flag, last_processed_text
    try:
        logger.info(f"Received message: {data}")
        new_text = data['text']
        language = data['language']
        speaker_wav = data['speaker_wav']

        if not speaker_wav:
            raise ValueError("No speaker file selected")

        speaker_wav_path = os.path.join(SPEAKERS_FOLDER, f"{speaker_wav}.wav")
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"Speaker file not found: {speaker_wav_path}")

        with queue_lock:
            # Process only the new text
            if last_processed_text and new_text.startswith(last_processed_text):
                text_to_process = new_text[len(last_processed_text):].strip()
            else:
                text_to_process = new_text

            # Preprocess the new text
            processed_text = preprocess_text(text_to_process)
            
            logger.info(f"Text to process after preprocessing: '{processed_text}'")
            
            if processed_text:
                text_buffer += processed_text
                if len(processed_text.split()) >= MIN_SENTENCE_LENGTH:
                    sentences = split_into_sentences(processed_text)
                else:
                    sentences = [processed_text]
                
                logger.info(f"Sentences: {sentences}")
                for sentence in sentences:
                    sentence_queue.append((sentence, speaker_wav_path, language))

                correction_flag = False
                last_processed_text = new_text

        logger.info("Calling process_sentence_queue")
        process_sentence_queue()

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        emit('error', {'error': str(e)})

    if time.time() % MEMORY_CHECK_INTERVAL < 1:
        check_memory_usage()

def process_sentence_queue():
    global paused, sentence_queue, correction_flag
    logger.info(f"Starting process_sentence_queue. Queue length: {len(sentence_queue)}")
    while sentence_queue:
        with queue_lock:
            if correction_flag:
                logger.info("Correction flag set, breaking loop")
                break
            sentence, speaker_wav_path, language = sentence_queue.popleft()
            logger.info(f"Processing sentence: '{sentence}'")

        if not paused:
            logger.info("Calling process_text")
            audio, processing_time, real_time_factor = process_text(sentence, speaker_wav_path, language)
            logger.info(f"process_text returned. Audio length: {len(audio)}")
            if len(audio) > 0:
                buffer = io.BytesIO()
                sf.write(buffer, audio, SAMPLE_RATE, format='WAV')
                audio_bytes = buffer.getvalue()

                logger.info("Emitting audio")
                emit('audio', {
                    'audio': audio_bytes,
                    'processing_time': processing_time,
                    'real_time_factor': real_time_factor
                }, broadcast=True)

                time.sleep(PROCESSING_DELAY)
            else:
                logger.warning("Audio length is 0, not emitting")
        else:
            logger.info("Speech generation is paused")
    logger.info("Finished process_sentence_queue")

if __name__ == '__main__':
    socketio.run(app, debug=True)