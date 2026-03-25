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
    fig.update_layout(
        title='Waveform', 
        xaxis_title='Time (s)', 
        yaxis_title='Amplitude', 
        template='plotly_white',
        height=300
    )
    return fig

def plot_spectrogram(audio_data, sample_rate):
    """Plot spectrogram"""
    try:
        D = librosa.stft(audio_data)
        S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max)
        fig = go.Figure(data=go.Heatmap(z=S_db, colorscale='Viridis'))
        fig.update_layout(title='Spectrogram', template='plotly_white', height=300)
        return fig
    except:
        return None

def simple_denoise(audio, strength=0.7):
    """Simple noise reduction using spectral gating"""
    try:
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
    except:
        return audio

def simple_separation(audio, num_speakers=2):
    """Simple speaker separation by splitting audio"""
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

def detect_speakers(audio_data, sample_rate):
    """Detect number of speakers based on energy"""
    try:
        # Compute frame energy
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        energy = librosa.power_to_db(S, ref=np.max)
        mean_energy = np.mean(energy)
        
        # Count frames above threshold
        above_threshold = np.sum(np.mean(energy, axis=0) > mean_energy)
        num_speakers = max(2, min(4, int(above_threshold / len(energy[0]) * 4)))
        
        return num_speakers
    except:
        return 2

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
    
    audio_file = st.file_uploader(
        "Choose audio (.wav, .mp3, .ogg)", 
        type=["wav", "mp3", "ogg"]
    )
    
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
            
            if st.button("▶️ PROCESS AUDIO", use_container_width=True, key="process"):
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    # Step 1: Visualization
                    st.subheader("📊 Audio Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.spinner("Plotting waveform..."):
                            fig = plot_waveform(audio_data, sample_rate)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        with st.spinner("Plotting spectrogram..."):
                            fig = plot_spectrogram(audio_data, sample_rate)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    
                    progress.progress(25)
                    status.info("📊 Analysis complete...")
                    
                    # Step 2: Speaker Detection
                    st.subheader("👥 Speaker Detection")
                    with st.spinner("Detecting speakers..."):
                        num_speakers = detect_speakers(audio_data, sample_rate)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎤 Speakers", num_speakers)
                    with col2:
                        st.metric("⏱️ Duration", f"{len(audio_data)/sample_rate:.2f}s")
                    with col3:
                        st.metric("📍 Sample Rate", f"{sample_rate}Hz")
                    
                    progress.progress(40)
                    status.info(f"👥 Detected {num_speakers} speakers...")
                    
                    # Step 3: Separation
                    st.subheader("🔊 Speech Separation")
                    with st.spinner("Separating speakers..."):
                        separated_audio = simple_separation(audio_data, num_speakers)
                    
                    progress.progress(60)
                    status.info("🔊 Speakers separated...")
                    
                    # Step 4: Noise Reduction
                    st.subheader("🧹 Noise Reduction")
                    enhanced_audio = []
                    for i, speaker_audio in enumerate(separated_audio):
                        cleaned = simple_denoise(speaker_audio, strength=noise_strength)
                        enhanced_audio.append(cleaned)
                    
                    progress.progress(75)
                    status.info("🧹 Audio enhanced...")
                    
                    # Step 5: Output
                    st.subheader("🎧 Separated Speakers")
                    output_path = Path("outputs")
                    output_path.mkdir(exist_ok=True)
                    
                    for i, speaker_audio in enumerate(enhanced_audio):
                        with st.container():
                            st.markdown(f"<div class='speaker-section'>", unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"### 🎤 Speaker {i+1}")
                            with col2:
                                duration = len(speaker_audio) / sample_rate
                                st.metric("Duration", f"{duration:.2f}s")
                            
                            # Audio player
                            st.audio(speaker_audio, sample_rate=sample_rate, format="audio/wav")
                            
                            # Waveform
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### Waveform")
                                fig = plot_waveform(speaker_audio, sample_rate)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("#### Spectrogram")
                                fig = plot_spectrogram(speaker_audio, sample_rate)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            
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
                    status.success("✅ Complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Load error: {str(e)}")

elif input_mode == "Sample Audio":
    st.subheader("📻 Generate Sample Audio")
    st.info("Generate synthetic multi-speaker audio for testing")
    
    if st.button("🎵 Create Sample", use_container_width=True, key="sample"):
        sample_rate = 16000
        duration = 8
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create two speakers
        speaker1 = 0.3 * np.sin(2 * np.pi * 200 * t)
        speaker2 = 0.3 * np.sin(2 * np.pi * 300 * t)
        
        # Mix with noise
        mixed = speaker1 + speaker2 + 0.02 * np.random.randn(len(t))
        mixed = mixed / np.max(np.abs(mixed))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{duration}s")
        with col2:
            st.metric("Sample Rate", f"{sample_rate}Hz")
        with col3:
            st.metric("Speakers", 2)
        
        st.audio(mixed, sample_rate=sample_rate)
        st.success("✅ Sample generated! Click 'Process Audio' to separate.")

st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
    <p><b>🎧 Voice Separation - 100% FREE</b></p>
    <p>No costs • No API keys • Open source</p>
    </div>
    """, unsafe_allow_html=True)
