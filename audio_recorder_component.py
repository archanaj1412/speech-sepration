import streamlit as st
import streamlit.components.v1 as components
import base64
import numpy as np
from scipy.io import wavfile
import io

def audio_recorder_html():
    """HTML/JS for browser audio recording using Web Audio API"""
    return """
    <html>
    <head>
        <style>
            body { font-family: Arial; padding: 20px; }
            .controls { margin-bottom: 20px; }
            button {
                padding: 10px 20px;
                margin: 5px;
                font-size: 16px;
                cursor: pointer;
                border: none;
                border-radius: 5px;
                background: #667eea;
                color: white;
            }
            button:hover { background: #764ba2; }
            button:disabled { background: #ccc; }
            #status { margin-top: 10px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="controls">
            <h3>🎙️ Browser Audio Recorder</h3>
            <button id="startBtn">🔴 Start Recording</button>
            <button id="stopBtn" disabled>⏹️ Stop Recording</button>
            <button id="resetBtn">🔄 Reset</button>
            <div id="status">Ready</div>
        </div>
        <audio id="playback" controls style="width: 100%; margin-top: 20px;"></audio>

        <script>
        let mediaRecorder;
        let audioChunks = [];
        let recordingStartTime;

        document.getElementById('startBtn').onclick = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                recordingStartTime = Date.now();
                
                mediaRecorder.ondataavailable = (e) => {
                    audioChunks.push(e.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        const base64Audio = btoa(
                            new Uint8Array(e.target.result)
                                .reduce((data, byte) => data + String.fromCharCode(byte), '')
                        );
                        
                        // Send to parent (Streamlit)
                        window.parent.postMessage({
                            type: 'audio_data',
                            audio: base64Audio,
                            duration: (Date.now() - recordingStartTime) / 1000
                        }, '*');
                        
                        // Play back
                        document.getElementById('playback').src = 'data:audio/wav;base64,' + base64Audio;
                    };
                    reader.readAsArrayBuffer(audioBlob);
                    
                    // Stop stream
                    stream.getTracks().forEach(track => track.stop());
                };
                
                mediaRecorder.start();
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('status').innerText = '🔴 Recording...';
                document.getElementById('status').style.color = 'red';
                
            } catch (err) {
                alert('Microphone access denied: ' + err.message);
                document.getElementById('status').innerText = '❌ Microphone error';
            }
        };

        document.getElementById('stopBtn').onclick = () => {
            mediaRecorder.stop();
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').innerText = '✅ Recording stopped. Playback available above.';
            document.getElementById('status').style.color = 'green';
        };

        document.getElementById('resetBtn').onclick = () => {
            audioChunks = [];
            document.getElementById('playback').src = '';
            document.getElementById('status').innerText = 'Ready';
            document.getElementById('status').style.color = 'black';
        };
        </script>
    </body>
    </html>
    """

def custom_audio_recorder():
    """Custom Streamlit component for audio recording"""
    components.html(audio_recorder_html(), height=400)

# Usage in Streamlit app
if __name__ == "__main__":
    st.title("Custom Web Audio Recorder")
    custom_audio_recorder()