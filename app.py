import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample
import warnings
warnings.filterwarnings('ignore')
from streamlit_audio_recorder import audio_recorder

# Import custom modules
from diarization import SpeakerDiarization
from separation import SpeechSeparation
from noise_reduction import NoiseReduction
from transcription import SpeechToText
from utils.helpers import (
    load_audio, save_audio, compute_metrics, 
    detect_silence, detect_overlap, create_visualizations
)

# Page configuration
st.set_page_config(
    page_title="AI Voice Separation System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .speaker-section {
        border-left: 4px solid #667eea;
        padding: 20px;
        margin: 15px 0;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'audio_loaded' not in st.session_state:
    st.session_state.audio_loaded = False
if 'separation_done' not in st.session_state:
    st.session_state.separation_done = False
if 'num_speakers' not in st.session_state:
    st.session_state.num_speakers = 0

# Sidebar configuration
st.sidebar.title("🎤 Voice Separation Control Panel")
st.sidebar.markdown("---")

# Input mode selection
input_mode = st.sidebar.radio(
    "📥 Select Input Mode:",
    ["Upload Audio File", "Record Audio (Live)", "Real-time Processing"],
    help="Choose how to input audio"
)

# Model selection
st.sidebar.markdown("### ⚙️ Model Settings")
separation_model = st.sidebar.selectbox(
    "Speech Separation Model",
    ["SepFormer", "Conv-TasNet", "Demucs"],
    help="Select the deep learning model for separation"
)

diarization_model = st.sidebar.selectbox(
    "Diarization Model",
    ["ECAPA-TDNN", "VoxCeleb"],
    help="Select speaker diarization model"
)

# Advanced options
with st.sidebar.expander("🔧 Advanced Settings"):
    noise_reduction_strength = st.slider(
        "Noise Reduction Strength",
        min_value=0.0, max_value=1.0, value=0.7
    )
    apply_enhancement = st.checkbox("Voice Enhancement", value=True)
    min_speaker_duration = st.slider(
        "Min Speaker Duration (seconds)",
        min_value=0.5, max_value=5.0, value=1.0
    )

# Main content area
st.title("🎧 AI-Powered Multi-Speaker Voice Separation System")
st.markdown("*Automatically detect, separate, and enhance multiple speakers with noise reduction*")

# Load audio based on input mode
audio_data = None
sample_rate = None
audio_file = None

if input_mode == "Upload Audio File":
    st.subheader("📤 Upload Audio File")
    audio_file = st.file_uploader(
        "Choose an audio file (.wav, .mp3, .ogg)",
        type=["wav", "mp3", "ogg"]
    )
    if audio_file:
        try:
            audio_data, sample_rate = librosa.load(audio_file, sr=None)
            st.session_state.audio_loaded = True
            st.success(f"✅ Audio loaded successfully! (Sample rate: {sample_rate} Hz, Duration: {len(audio_data)/sample_rate:.2f}s)")
        except Exception as e:
            st.error(f"❌ Error loading audio: {str(e)}")

elif input_mode == "Record Audio (Live)":
    st.subheader("🎙️ Record Audio - Live Microphone Input")
    st.markdown("**Click the record button below to start recording from your microphone**")
    
    # Audio recorder component (NO FFmpeg needed!)
    audio_bytes = audio_recorder(
        text="🎙️ Click to record",
        pause_threshold=2.0,
        sample_rate=16000,
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # Convert bytes to numpy array
        import io
        audio_data, sample_rate = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=None,
            mono=True
        )
        
        st.session_state.audio_loaded = True
        st.success(f"✅ Audio recorded successfully! (Duration: {len(audio_data)/sample_rate:.2f}s)")

elif input_mode == "Real-time Processing":
    st.subheader("📡 Real-time Audio Processing")
    st.info("Real-time processing with chunked audio processing")
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk Size (seconds)", 2, 10, 5)
    with col2:
        overlap = st.slider("Overlap (seconds)", 0, 3, 1)
    
    # Record audio for real-time processing
    audio_bytes = audio_recorder(
        text="🔴 Start Real-time Recording",
        pause_threshold=3.0,
        sample_rate=16000,
    )
    
    if audio_bytes:
        import io
        audio_data, sample_rate = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=None,
            mono=True
        )
        st.session_state.audio_loaded = True
        st.success(f"✅ Audio recorded for real-time processing! (Duration: {len(audio_data)/sample_rate:.2f}s)")

# Process audio if loaded
if st.session_state.audio_loaded and audio_data is not None:
    st.markdown("---")
    
    # Create progress tracker
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Visualization of raw audio
        st.subheader("📊 Raw Audio Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Waveform")
            fig_wave = create_visualizations.plot_waveform(audio_data, sample_rate)
            st.plotly_chart(fig_wave, use_container_width=True)
        
        with col2:
            st.markdown("#### Spectrogram")
            fig_spec = create_visualizations.plot_spectrogram(audio_data, sample_rate)
            st.plotly_chart(fig_spec, use_container_width=True)
        
        progress_bar.progress(10)
        status_text.info("📊 Visualization complete")
        
        # Step 2: Noise Analysis
        st.subheader("🔇 Noise Analysis")
        silence_frames = detect_silence(audio_data, sample_rate)
        silence_percentage = (np.sum(silence_frames) / len(silence_frames)) * 100
        st.metric("Silence Percentage", f"{silence_percentage:.2f}%")
        
        progress_bar.progress(20)
        status_text.info("🔇 Analyzing noise profile...")
        
        # Step 3: Speaker Diarization
        st.subheader("👥 Speaker Detection & Diarization")
        
        diarization = SpeakerDiarization(model_name=diarization_model)
        num_speakers, speaker_intervals = diarization.detect_speakers(
            audio_data, 
            sample_rate,
            min_duration=0.5
        )
        st.session_state.num_speakers = num_speakers
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎤 Speakers Detected", num_speakers)
        with col2:
            st.metric("⏱️ Audio Duration", f"{len(audio_data)/sample_rate:.2f}s")
        with col3:
            st.metric("📍 Sample Rate", f"{sample_rate} Hz")
        
        progress_bar.progress(40)
        status_text.info(f"👥 Detected {num_speakers} speakers")
        
        # Step 4: Speech Separation
        st.subheader("🔊 Speech Separation")
        
        separator = SpeechSeparation(model_name=separation_model)
        separated_audio = separator.separate(
            audio_data, 
            sample_rate, 
            num_speakers=num_speakers
        )
        
        progress_bar.progress(60)
        status_text.info("🔊 Separating speakers...")
        
        # Step 5: Noise Reduction
        st.subheader("🧹 Noise Reduction & Enhancement")
        
        noise_reducer = NoiseReduction()
        enhanced_audio = []
        
        for i, speaker_audio in enumerate(separated_audio):
            cleaned = noise_reducer.reduce_noise(
                speaker_audio,
                sample_rate,
                strength=noise_reduction_strength
            )
            
            if apply_enhancement:
                cleaned = noise_reducer.enhance_voice(cleaned, sample_rate)
            
            enhanced_audio.append(cleaned)
        
        progress_bar.progress(70)
        status_text.info("🧹 Reducing noise and enhancing clarity...")
        
        # Step 6: Audio Quality Metrics
        st.subheader("📈 Audio Quality Metrics")
        
        metrics_data = []
        for i, speaker_audio in enumerate(enhanced_audio):
            snr, sdr = compute_metrics(separated_audio[i], speaker_audio)
            metrics_data.append({
                "Speaker": f"Speaker {i+1}",
                "SNR (dB)": snr,
                "SDR (dB)": sdr
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        progress_bar.progress(80)
        status_text.info("📈 Computing quality metrics...")
        
        # Step 7: Transcription
        st.subheader("🗣️ Speech-to-Text Transcription")
        
        transcriber = SpeechToText()
        transcriptions = []
        
        with st.spinner("Transcribing audio..."):
            for i, speaker_audio in enumerate(enhanced_audio):
                text = transcriber.transcribe(speaker_audio, sample_rate)
                language = transcriber.detect_language(text)
                transcriptions.append({
                    "speaker": i + 1,
                    "text": text,
                    "language": language
                })
        
        progress_bar.progress(85)
        status_text.info("🗣️ Transcribing speakers...")
        
        # Step 8: Display separated speakers
        st.subheader("🎧 Separated Speaker Outputs")
        
        st.session_state.separation_done = True
        
        for i, (speaker_audio, transcription) in enumerate(zip(enhanced_audio, transcriptions)):
            with st.container():
                st.markdown(f"<div class='speaker-section'>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### 🎤 Speaker {i+1}")
                
                with col2:
                    st.markdown(f"**Language:** {transcription['language']}")
                
                with col3:
                    speaker_silence = np.mean(np.abs(speaker_audio)) < 0.01
                    if speaker_silence:
                        st.markdown("⚪ Silence")
                    else:
                        st.markdown("🟢 Active")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SNR", f"{metrics_df.iloc[i]['SNR (dB)']:.2f} dB")
                with col2:
                    st.metric("SDR", f"{metrics_df.iloc[i]['SDR (dB)']:.2f} dB")
                
                st.audio(speaker_audio, sample_rate=sample_rate, format="audio/wav")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Waveform")
                    fig_wave = create_visualizations.plot_waveform(speaker_audio, sample_rate)
                    st.plotly_chart(fig_wave, use_container_width=True)
                
                with col2:
                    st.markdown("#### Spectrogram")
                    fig_spec = create_visualizations.plot_spectrogram(speaker_audio, sample_rate)
                    st.plotly_chart(fig_spec, use_container_width=True)
                
                st.markdown("#### 📝 Transcription")
                st.text_area(
                    f"Text for Speaker {i+1}",
                    value=transcription['text'],
                    height=100,
                    disabled=True,
                    key=f"transcription_{i}"
                )
                
                output_path = Path("outputs") / f"speaker_{i+1}.wav"
                output_path.parent.mkdir(exist_ok=True)
                sf.write(str(output_path), speaker_audio, sample_rate)
                
                with open(str(output_path), "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download Speaker {i+1}",
                        data=f.read(),
                        file_name=f"speaker_{i+1}.wav",
                        mime="audio/wav",
                        key=f"download_{i}"
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        progress_bar.progress(90)
        status_text.info("✅ Processing complete!")
        
        # Step 9: Logging & Reporting
        st.subheader("📋 Processing Report")
        
        report_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Input Duration (s)": len(audio_data) / sample_rate,
            "Sample Rate (Hz)": sample_rate,
            "Speakers Detected": num_speakers,
            "Diarization Model": diarization_model,
            "Separation Model": separation_model,
            "Noise Reduction Strength": noise_reduction_strength
        }
        
        report_df = pd.DataFrame([report_data])
        st.dataframe(report_df, use_container_width=True)
        
        logs_path = Path("logs")
        logs_path.mkdir(exist_ok=True)
        report_csv = logs_path / "results.csv"
        
        if report_csv.exists():
            existing_df = pd.read_csv(report_csv)
            report_df = pd.concat([existing_df, report_df], ignore_index=True)
        
        report_df.to_csv(str(report_csv), index=False)
        
        csv_data = report_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processing Report",
            data=csv_data,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        progress_bar.progress(100)
        status_text.success("✅ All processing complete!")
        
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
    <p>🎧 <b>AI Voice Separation System v1.0</b></p>
    <p>Powered by SepFormer, ECAPA-TDNN, and Whisper AI</p>
    <p><small>No FFmpeg Required - Uses Browser Web Audio API</small></p>
    </div>
    """, unsafe_allow_html=True)