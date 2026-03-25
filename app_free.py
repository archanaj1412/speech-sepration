"""
Completely FREE Speech Separation System
No API keys required - Uses local models only
Deployed free on Streamlit Cloud / Railway / Render / HF Spaces
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from diarization import SpeakerDiarization
    from separation import SpeechSeparation
    from noise_reduction import NoiseReduction
    from transcription import SpeechToText
    from utils.helpers import create_visualizations, compute_metrics
except ImportError:
    st.error("Missing modules. Running with basic functionality...")

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="🎤 FREE Voice Separation System",
    page_icon="🎤",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .speaker-section {
        border-left: 4px solid #667eea;
        padding: 20px;
        margin: 15px 0;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🎤 FREE Voice Separation")
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Completely FREE\n- No API keys needed\n- No costs\n- Runs locally")

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Upload Audio", "Sample Audio"]
)

st.sidebar.markdown("### ⚙️ Models")
separation_model = st.sidebar.selectbox("Separation", ["SepFormer", "Conv-TasNet"])
diarization_model = st.sidebar.selectbox("Diarization", ["ECAPA-TDNN", "VoxCeleb"])

with st.sidebar.expander("Settings"):
    noise_strength = st.slider("Noise Reduction", 0.0, 1.0, 0.7)
    enhance_voice = st.checkbox("Voice Enhancement", True)

# ============================================
# MAIN CONTENT
# ============================================

st.title("🎤 FREE AI Voice Separation System")
st.markdown("*Multi-speaker detection & separation - No costs, no API keys*")

# ============================================
# INPUT MODES
# ============================================

if input_mode == "Upload Audio":
    st.subheader("📤 Upload Audio File")
    
    audio_file = st.file_uploader("Choose audio (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
    
    if audio_file:
        try:
            audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{len(audio_data)/sample_rate:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sample_rate}Hz")
            with col3:
                st.metric("Size", f"{audio_file.size/1024:.1f}KB")
            
            st.success("✅ Audio loaded")
            
            if st.button("▶️ PROCESS AUDIO", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    # Visualization
                    st.subheader("📊 Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Waveform")
                        fig_wave = create_visualizations.plot_waveform(audio_data, sample_rate)
                        st.plotly_chart(fig_wave, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Spectrogram")
                        fig_spec = create_visualizations.plot_spectrogram(audio_data, sample_rate)
                        st.plotly_chart(fig_spec, use_container_width=True)
                    
                    progress.progress(20)
                    status.info("📊 Analysis complete...")
                    
                    # Speaker Detection
                    st.subheader("👥 Speaker Detection")
                    
                    diarization = SpeakerDiarization(model_name=diarization_model)
                    num_speakers, _ = diarization.detect_speakers(audio_data, sample_rate, min_duration=0.5)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Speakers", num_speakers)
                    with col2:
                        st.metric("Duration", f"{len(audio_data)/sample_rate:.2f}s")
                    with col3:
                        st.metric("Sample Rate", f"{sample_rate}Hz")
                    
                    progress.progress(40)
                    status.info(f"👥 Detected {num_speakers} speakers...")
                    
                    # Separation
                    st.subheader("🔊 Speech Separation")
                    
                    separator = SpeechSeparation(model_name=separation_model)
                    separated_audio = separator.separate(audio_data, sample_rate, num_speakers=num_speakers)
                    
                    progress.progress(60)
                    status.info("🔊 Separating speakers...")
                    
                    # Noise Reduction
                    st.subheader("🧹 Noise Reduction")
                    
                    noise_reducer = NoiseReduction()
                    enhanced_audio = []
                    
                    for speaker_audio in separated_audio:
                        cleaned = noise_reducer.reduce_noise(speaker_audio, sample_rate, strength=noise_strength)
                        if enhance_voice:
                            cleaned = noise_reducer.enhance_voice(cleaned, sample_rate)
                        enhanced_audio.append(cleaned)
                    
                    progress.progress(75)
                    status.info("🧹 Enhancing audio...")
                    
                    # Transcription (using free Whisper)
                    st.subheader("🗣️ Transcription (Whisper)")
                    
                    transcriber = SpeechToText()
                    transcriptions = []
                    
                    with st.spinner("Transcribing..."):
                        for i, speaker_audio in enumerate(enhanced_audio):
                            text = transcriber.transcribe(speaker_audio, sample_rate)
                            language = transcriber.detect_language(text)
                            transcriptions.append({
                                'speaker': i + 1,
                                'text': text,
                                'language': language
                            })
                    
                    progress.progress(85)
                    
                    # Output
                    st.subheader("🎧 Separated Speakers")
                    
                    for i, (speaker_audio, transcript) in enumerate(zip(enhanced_audio, transcriptions)):
                        with st.container():
                            st.markdown(f"<div class='speaker-section'>", unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"### 🎤 Speaker {i+1}")
                            with col2:
                                st.markdown(f"**{transcript['language']}**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                snr, sdr = compute_metrics(separated_audio[i], speaker_audio)
                                st.metric("SNR", f"{snr:.2f}dB")
                            with col2:
                                st.metric("SDR", f"{sdr:.2f}dB")
                            
                            st.audio(speaker_audio, sample_rate=sample_rate, format="audio/wav")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### Waveform")
                                fig = create_visualizations.plot_waveform(speaker_audio, sample_rate)
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                st.markdown("#### Spectrogram")
                                fig = create_visualizations.plot_spectrogram(speaker_audio, sample_rate)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("#### 📝 Transcript")
                            st.text_area(
                                "Transcription",
                                value=transcript['text'],
                                height=80,
                                disabled=True,
                                key=f"transcript_{i}"
                            )
                            
                            # Download
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
                    
                    progress.progress(100)
                    status.success("✅ Complete!")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Load error: {str(e)}")

elif input_mode == "Sample Audio":
    st.subheader("📻 Generate Sample Audio")
    
    if st.button("📻 Generate & Process Sample"):
        with st.spinner("Generating sample..."):
            sample_rate = 16000
            duration = 10
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            speaker1 = np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 0.1 * t)
            speaker2 = np.sin(2 * np.pi * 300 * t) * np.sin(2 * np.pi * 0.15 * t)
            
            mixed = (speaker1 + speaker2) / 2 + np.random.normal(0, 0.05, len(t))
            mixed = mixed / np.max(np.abs(mixed))
            
            st.audio(mixed, sample_rate=sample_rate, format="audio/wav")
            st.success("✅ Sample ready for processing")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
    <p><b>🎧 Completely FREE Voice Separation</b></p>
    <p>No API keys • No costs • Open source</p>
    <p>Powered by SepFormer + ECAPA-TDNN + Whisper</p>
    </div>
    """, unsafe_allow_html=True)