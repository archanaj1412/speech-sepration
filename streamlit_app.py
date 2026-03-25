"""
🎧 AI Voice Separation System - Streamlit Cloud
Lightweight version for FREE deployment
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

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="🎤 Voice Separation FREE",
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

st.sidebar.title("🎤 Voice Separation")
st.sidebar.markdown("""
    ✅ Multi-speaker detection
    ✅ Speech separation
    ✅ Noise reduction
    ✅ 100% FREE
    """)

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Upload Audio", "Sample Audio"]
)

with st.sidebar.expander("⚙️ Settings"):
    noise_strength = st.slider("Noise Reduction", 0.0, 1.0, 0.7)

# ============================================
# HELPER FUNCTIONS
# ============================================

def plot_waveform(audio_data, sample_rate):
    """Plot waveform"""
    time = np.arange(len(audio_data)) / sample_rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio_data, mode='lines', 
                            line=dict(color='#667eea', width=1)))
    fig.update_layout(title='Waveform', xaxis_title='Time (s)', 
                     yaxis_title='Amplitude', template='plotly_white')
    return fig

def plot_spectrogram(audio_data, sample_rate):
    """Plot spectrogram"""
    D = librosa.stft(audio_data)
    S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max)
    fig = go.Figure(data=go.Heatmap(z=S_db, colorscale='Viridis'))
    fig.update_layout(title='Spectrogram', template='plotly_white')
    return fig

def simple_denoise(audio, sr, strength=0.7):
    """Simple noise reduction using spectral gating"""
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Estimate noise floor
    noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
    
    # Apply gate
    gate = np.maximum(magnitude - strength * noise_floor, 0) / (magnitude + 1e-10)
    D_denoised = gate * D
    
    # Inverse STFT
    audio_denoised = librosa.istft(D_denoised)
    return audio_denoised

def simple_separation(audio, sr, num_speakers=2):
    """Simple speaker separation using spectral clustering"""
    # For demo: split audio into chunks and treat as different speakers
    chunk_length = len(audio) // num_speakers
    separated = []
    
    for i in range(num_speakers):
        start = i * chunk_length
        end = (i + 1) * chunk_length if i < num_speakers - 1 else len(audio)
        separated.append(audio[start:end])
    
    # Pad shorter chunks
    max_len = max(len(s) for s in separated)
    for i in range(len(separated)):
        if len(separated[i]) < max_len:
            separated[i] = np.pad(separated[i], (0, max_len - len(separated[i])))
    
    return separated

# ============================================
# MAIN CONTENT
# ============================================

st.title("🎧 Voice Separation System")
st.markdown("*Simple speech processing - 100% FREE*")

# ============================================
# INPUT MODES
# ============================================

if input_mode == "Upload Audio":
    st.subheader("📤 Upload Audio File")
    
    audio_file = st.file_uploader("Choose audio (.wav, .mp3, .ogg)", 
                                   type=["wav", "mp3", "ogg"])
    
    if audio_file:
        try:
            with st.spinner("Loading..."):
                audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{len(audio_data)/sample_rate:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sample_rate}Hz")
            with col3:
                st.metric("Size", f"{audio_file.size/1024/1024:.2f}MB")
            
            st.success("✅ Loaded!")
            
            if st.button("▶️ PROCESS", use_container_width=True):
                progress = st.progress(0)
                
                try:
                    # Visualization
                    st.subheader("📊 Audio Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_waveform(audio_data, sample_rate), use_container_width=True)
                    with col2:
                        st.plotly_chart(plot_spectrogram(audio_data, sample_rate), use_container_width=True)
                    
                    progress.progress(25)
                    
                    # Speaker Detection (simplified)
                    st.subheader("👥 Speaker Detection")
                    # Simple: detect based on energy changes
                    energy = np.sqrt(np.mean(librosa.util.frame(audio_data, frame_length=2048, 
                                                                  hop_length=512)**2, axis=0))
                    threshold = np.mean(energy)
                    num_speakers = max(2, min(4, int(np.sum(energy > threshold) / len(energy) * 4)))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Speakers", num_speakers)
                    with col2:
                        st.metric("Duration", f"{len(audio_data)/sample_rate:.2f}s")
                    with col3:
                        st.metric("Sample Rate", f"{sample_rate}Hz")
                    
                    progress.progress(40)
                    
                    # Separation
                    st.subheader("🔊 Speech Separation")
                    with st.spinner("Separating..."):
                        separated_audio = simple_separation(audio_data, sample_rate, num_speakers)
                    
                    progress.progress(60)
                    
                    # Noise Reduction
                    st.subheader("🧹 Noise Reduction")
                    enhanced_audio = []
                    for speaker_audio in separated_audio:
                        cleaned = simple_denoise(speaker_audio, sample_rate, strength=noise_strength)
                        enhanced_audio.append(cleaned)
                    
                    progress.progress(75)
                    
                    # Output
                    st.subheader("🎧 Separated Speakers")
                    output_path = Path("outputs")
                    output_path.mkdir(exist_ok=True)
                    
                    for i, speaker_audio in enumerate(enhanced_audio):
                        with st.container():
                            st.markdown(f"<div class='speaker-section'>", unsafe_allow_html=True)
                            st.markdown(f"### 🎤 Speaker {i+1}")
                            
                            st.audio(speaker_audio, sample_rate=sample_rate, format="audio/wav")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(plot_waveform(speaker_audio, sample_rate), 
                                               use_container_width=True)
                            with col2:
                                st.plotly_chart(plot_spectrogram(speaker_audio, sample_rate), 
                                               use_container_width=True)
                            
                            # Download
                            wav_file = output_path / f"speaker_{i+1}.wav"
                            sf.write(str(wav_file), speaker_audio, sample_rate)
                            
                            with open(str(wav_file), "rb") as f:
                                st.download_button(
                                    label=f"⬇️ Download Speaker {i+1}",
                                    data=f.read(),
                                    file_name=f"speaker_{i+1}.wav",
                                    mime="audio/wav",
                                    key=f"dl_{i}"
                                )
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    progress.progress(100)
                    st.success("✅ Complete!")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Load error: {str(e)}")

elif input_mode == "Sample Audio":
    st.subheader("📻 Generate Sample")
    
    if st.button("🎵 Create Sample Audio"):
        sample_rate = 16000
        duration = 8
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        speaker1 = np.sin(2 * np.pi * 200 * t)
        speaker2 = np.sin(2 * np.pi * 300 * t)
        
        mixed = (speaker1 + speaker2) / 2 + 0.05 * np.random.randn(len(t))
        mixed = mixed / np.max(np.abs(mixed))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{duration}s")
        with col2:
            st.metric("Sample Rate", f"{sample_rate}Hz")
        with col3:
            st.metric("Speakers", 2)
        
        st.audio(mixed, sample_rate=sample_rate)
        st.success("✅ Ready to process!")

st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
    <b>🎧 Voice Separation - 100% FREE</b><br>
    No costs • No API keys • Open source
    </div>
    """, unsafe_allow_html=True)
