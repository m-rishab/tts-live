document.addEventListener('DOMContentLoaded', () => {
    let socket = null;
    const connectBtn = document.getElementById('connectBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');
    const textInput = document.getElementById('textInput');
    const audioOutput = document.getElementById('audioOutput');
    const languageSelect = document.getElementById('language');
    const speakerSelect = document.getElementById('speaker');
    const speakerFileInput = document.getElementById('speakerFile');
    const uploadBtn = document.getElementById('uploadBtn');
    let isConnected = false;
    let typingTimer;
    const doneTypingInterval = 300; // ms
    const audioQueue = [];
    let isPlaying = false;
    let audioContext;
    let lastProcessedText = '';

    // Function to update speakers
    function updateSpeakers() {
        fetch(`/speakers/`, {
            method: 'GET',
            headers: {
                "Content-Type": "application/json"
            }
        })
        .then(response => response.json())
        .then(speakers => {
            speakerSelect.innerHTML = '';
            speakers.forEach(speaker => {
                const option = document.createElement('option');
                option.value = speaker;
                option.textContent = speaker;
                speakerSelect.appendChild(option);
            });
        });
    }

    // Event listener for speaker upload form submission
    uploadBtn.addEventListener('click', () => {
        const formData = new FormData();
        formData.append('speakerFile', speakerFileInput.files[0]);
        formData.append('language', languageSelect.value);
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(result => {
            alert(result.message);
            updateSpeakers();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('File upload failed');
        });
    });

    // Event listener for connect button
    connectBtn.addEventListener('click', () => {
        if (!isConnected) {
            socket = io();
            setupSocketEvents(socket);
            socket.connect();
            console.log('WebSocket connected');
        }
    });

    // Event listener for disconnect button
    disconnectBtn.addEventListener('click', () => {
        if (isConnected && socket) {
            socket.disconnect();
        }
    });

    // Event listener for text input change
    textInput.addEventListener('input', () => {
        clearTimeout(typingTimer);
        typingTimer = setTimeout(doneTyping, doneTypingInterval);
    });

    // Function to handle when user is done typing
    function doneTyping() {
        if (isConnected && socket) {
            const text = textInput.value;
            const language = languageSelect.value;
            const speaker = speakerSelect.value;
            
            // Send only the new text
            let newText = text;
            if (lastProcessedText && text.startsWith(lastProcessedText)) {
                newText = text.slice(lastProcessedText.length);
            }
            
            if (newText.trim()) {
                console.log('Sending text to server:', { language, speaker, text: newText });
                socket.emit('message', { language, speaker_wav: speaker, text: newText });
                lastProcessedText = text;
            }
        } else {
            console.log('Not connected or no socket available');
        }
    }

    // Function to set up socket events
    function setupSocketEvents(socket) {
        socket.on('connect', () => {
            isConnected = true;
            connectBtn.disabled = true;
            disconnectBtn.disabled = false;
            textInput.disabled = false;
            console.log('Connected to server');
            alert('WebSocket connected');
        });

        socket.on('disconnect', () => {
            isConnected = false;
            connectBtn.disabled = false;
            disconnectBtn.disabled = true;
            textInput.disabled = true;
            console.log('Disconnected from server');
            alert('WebSocket disconnected');
        });

        socket.on('audio', (data) => {
            console.log('Received audio data');
            const blob = new Blob([data.audio], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            audioQueue.push(url);
            if (!isPlaying) {
                playNextAudio();
            }
        });

        socket.on('response', (data) => {
            console.log(data.message);
        });

        socket.on('error', (data) => {
            console.error(data.error);
        });
    }

    function playNextAudio() {
        if (audioQueue.length > 0) {
            isPlaying = true;
            const url = audioQueue.shift();
            audioOutput.src = url;
            audioOutput.play();
            audioOutput.onended = () => {
                URL.revokeObjectURL(url);
                playNextAudio();
            };
        } else {
            isPlaying = false;
        }
    }

    // Initialize audio context after user interaction
    document.body.addEventListener('click', () => {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
    });

    // Initial update of speakers
    updateSpeakers();
});