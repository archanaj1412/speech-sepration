import streamlit as st
import asyncio
import pyaudio
import numpy as np
import requests
from datetime import datetime

# Option A: AssemblyAI Real-time
def transcribe_with_assemblyai():
    """Live transcription with AssemblyAI"""
    st.subheader("🎙️ Live Transcription with AssemblyAI")
    
    api_key = st.text_input("Enter AssemblyAI API Key", type="password")
    
    if api_key and st.button("Start Live Transcription"):
        import assemblyai as aai
        aai.settings.api_key = api_key
        
        with st.spinner("Transcribing..."):
            transcriber = aai.RealtimeTranscriber(
                on_data=lambda transcript: st.write(f"📝 {transcript.text}"),
                on_error=lambda error: st.error(f"Error: {error}"),
                on_end=lambda: st.success("Transcription complete!"),
                encoding=aai.AudioEncoding.pcm_16,
                sample_rate=16000
            )
            
            transcriber.connect()
            
            # Capture mic audio
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
            
            while transcriber.is_connected():
                data = stream.read(4096)
                transcriber.send(data)
            
            transcriber.close()
            stream.stop_stream()
            stream.close()
            p.terminate()

# Option B: OpenAI Whisper API (Realtime not available, but batch processing)
def transcribe_with_openai():
    """Transcription with OpenAI Whisper API"""
    st.subheader("🎙️ Transcription with OpenAI Whisper API")
    
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    if api_key:
        import openai
        openai.api_key = api_key
        
        if st.button("Record and Transcribe"):
            with st.spinner("Recording..."):
                import pyaudio
                CHUNK = 1024
                FORMAT = pyaudio.paFloat32
                CHANNELS = 1
                RATE = 16000
                RECORD_SECONDS = 5
                
                p = pyaudio.PyAudio()
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
                
                frames = []
                progress_bar = st.progress(0)
                
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)
                    progress_bar.progress((i + 1) / int(RATE / CHUNK * RECORD_SECONDS))
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                # Convert to WAV
                import wave
                with wave.open("temp_audio.wav", 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                
                # Transcribe with Whisper
                with st.spinner("Transcribing with OpenAI Whisper..."):
                    with open("temp_audio.wav", "rb") as audio_file:
                        transcript = openai.Audio.transcribe(
                            model="whisper-1",
                            file=audio_file
                        )
                    
                    st.success("✅ Transcription complete!")
                    st.write(f"📝 **Transcript:** {transcript['text']}")
                    
                    import os
                    os.remove("temp_audio.wav")

# Main app
st.title("🎧 Live Audio Streaming & Transcription")

option = st.radio("Choose transcription service:", 
    ["streamlit-audio-recorder (Easiest)", "AssemblyAI (Realtime)", "OpenAI Whisper API"])

if option == "streamlit-audio-recorder (Easiest)":
    from streamlit_audio_recorder import audio_recorder
    
    st.info("✅ **No external API needed!** Uses your microphone directly.")
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

elif option == "AssemblyAI (Realtime)":
    transcribe_with_assemblyai()

elif option == "OpenAI Whisper API":
    transcribe_with_openai()