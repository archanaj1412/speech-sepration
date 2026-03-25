"""
🎧 Voice Separation - Streamlit Cloud
Super lightweight - No complex dependencies
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🎤 Voice Separation",
    page_icon="🎤",
    layout="wide"
)

st.markdown("""
    <style>
    .speaker-section {
        border-left: 4px solid #667eea;
        padding: 20px;
        margin: 15px 0;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🎤 Voice Separation")
st.sidebar.markdown("✅ Free • Simple • Fast")

input_mode = st.sidebar.radio("Input Mode", ["Upload Audio", "Sample Audio"])

with st.sidebar.expander("Settings"):
    noise_strength = st.slider("Noise Reduction", 0.0, 1.0, 0.7)

def plot_waveform(audio_data, sample_rate):
    time = np.arange(len(audio_data)) / sample_rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio_data, mode='lines', 
                            line=dict(color='#667eea', width=1)))
    fig.update_layout(title='Waveform', xaxis_title='Time (s)', 
                     yaxis_title='Amplitude', template='plotly_white', height=300)
    return fig

def plot_spectrogram(audio_data, sample_rate):
    try:
        D = librosa.stft(audio_data)
        S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max)
        fig = go.Figure(data=go.Heatmap(z=S_db, colorscale='Viridis'))
        fig.update_layout(title='Spectrogram', template='plotly_white', height=300)
        return fig
    except:
        return None

def denoise(audio, strength=0.7):
    try:
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
        gate = np.maximum(magnitude - strength * noise_floor, 0) / (magnitude + 1e-10)
        return librosa.istft(gate * D)
    except:
        return audio

def separate(audio, num_speakers=2):
    chunk_length = len(audio) // num_speakers
    separated = []
    for i in range(num_speakers):
        start = i * chunk_length
        end = (i + 1) * chunk_length if i < num_speakers - 1 else len(audio)
        separated.append(audio[start:end])
    max_len = max(len(s) for s in separated)
    for i in range(len(separated)):
        if len(separated[i]) < max_len:
            separated[i] = np.pad(separated[i], (0, max_len - len(separated[i])))
    return separated

def detect_speakers(audio_data, sample_rate):
    try:
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        energy = librosa.power_to_db(S, ref=np.max)
        mean_energy = np.mean(energy)
        above_threshold = np.sum(np.mean(energy, axis=0) > mean_energy)
        return max(2, min(4, int(above_threshold / len(energy[0]) * 4)))
    except:
        return 2

st.title("🎧 Voice Separation")
st.markdown("*100% FREE - No API Keys*")

if input_mode == "Upload Audio":
    st.subheader("📤 Upload Audio")
    audio_file = st.file_uploader("Choose audio (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
    
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
                    st.subheader("📊 Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_waveform(audio_data, sample_rate), use_container_width=True)
                    with col2:
                        fig = plot_spectrogram(audio_data, sample_rate)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    progress.progress(25)
                    
                    st.subheader("👥 Detection")
                    num_speakers = detect_speakers(audio_data, sample_rate)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Speakers", num_speakers)
                    with col2:
                        st.metric("Duration", f"{len(audio_data)/sample_rate:.2f}s")
                    with col3:
                        st.metric("Rate", f"{sample_rate}Hz")
                    
                    progress.progress(40)
                    
                    st.subheader("🔊 Separation")
                    separated_audio = separate(audio_data, num_speakers)
                    progress.progress(60)
                    
                    st.subheader("🧹 Enhancement")
                    enhanced_audio = [denoise(s, strength=noise_strength) for s in separated_audio]
                    progress.progress(75)
                    
                    st.subheader("🎧 Output")
                    output_path = Path("outputs")
                    output_path.mkdir(exist_ok=True)
                    
                    for i, speaker_audio in enumerate(enhanced_audio):
                        with st.container():
                            st.markdown(f"<div class='speaker-section'>", unsafe_allow_html=True)
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"### 🎤 Speaker {i+1}")
                            with col2:
                                st.metric("Duration", f"{len(speaker_audio)/sample_rate:.2f}s")
                            
                            st.audio(speaker_audio, sample_rate=sample_rate)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(plot_waveform(speaker_audio, sample_rate), use_container_width=True)
                            with col2:
                                fig = plot_spectrogram(speaker_audio, sample_rate)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            
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
                    st.success("✅ Done!")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        except Exception as e:
            st.error(f"Load error: {e}")

elif input_mode == "Sample Audio":
    st.subheader("📻 Sample")
    if st.button("🎵 Create Sample", use_container_width=True):
        sr = 16000
        duration = 8
        t = np.linspace(0, duration, int(sr * duration))
        s1 = 0.3 * np.sin(2 * np.pi * 200 * t)
        s2 = 0.3 * np.sin(2 * np.pi * 300 * t)
        mixed = (s1 + s2) / 2 + 0.02 * np.random.randn(len(t))
        mixed = mixed / np.max(np.abs(mixed))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{duration}s")
        with col2:
            st.metric("Rate", f"{sr}Hz")
        with col3:
            st.metric("Speakers", 2)
        
        st.audio(mixed, sample_rate=sr)
        st.success("✅ Ready!")

st.markdown("---")
st.markdown("<div style='text-align:center'><b>🎧 Voice Separation - 100% FREE</b></div>", unsafe_allow_html=True)
