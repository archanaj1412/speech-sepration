"""
Complete AI Voice Separation System with AssemblyAI Real-Time Transcription
Production-ready deployment
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import json
import os
from io import BytesIO
import time

# Import custom modules
try:
    from diarization import SpeakerDiarization
    from separation import SpeechSeparation
    from noise_reduction import NoiseReduction
    from transcription import SpeechToText
    from assemblyai_realtime import RealtimeTranscriptionManager
    from utils.helpers import create_visualizations, detect_silence, compute_metrics
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="🎤 AI Voice Separation + Real-Time Transcription",
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
    .transcript-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
    }
    .status-live {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #ff4444;
        border-radius: 50%;
        animation: pulse 1s infinite;
        margin-right: 8px;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

session_defaults = {
    'audio_loaded': False,
    'separation_done': False,
    'num_speakers': 0,
    'realtime_active': False,
    'full_transcript': [],
    'speaker_transcripts': [],
    'api_key_valid': False,
    'audio_data': None,
    'sample_rate': None
}

for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================
# SIDEBAR CONFIGURATION
# ============================================

st.sidebar.title("🎤 Control Panel")
st.sidebar.markdown("---")

# 1. API Key Configuration
st.sidebar.markdown("### 🔑 API Configuration")
api_key = st.sidebar.text_input(
    "AssemblyAI API Key",
    type="password",
    help="Get free key: https://www.assemblyai.com"
)

if api_key:
    st.session_state.api_key_valid = True
    st.sidebar.success("✅ API Key configured")
else:
    st.sidebar.info("ℹ️ Optional: Enter API key for real-time transcription")

# 2. Input Mode Selection
input_mode = st.sidebar.radio(
    "📥 Input Mode",
    ["Upload File", "🔴 Live Transcription", "Sample Audio"]
)

# 3. Model Selection
st.sidebar.markdown("### ⚙️ Models")
col1, col2 = st.sidebar.columns(2)
with col1:
    separation_model = st.selectbox("Separation", ["SepFormer", "Conv-TasNet"])
with col2:
    diarization_model = st.selectbox("Diarization", ["ECAPA-TDNN", "VoxCeleb"])

# 4. Advanced Settings
with st.sidebar.expander("🔧 Settings", expanded=False):
    noise_strength = st.slider("Noise Reduction", 0.0, 1.0, 0.7, 0.1)
    enhance_voice = st.checkbox("Voice Enhancement", True)
    min_duration = st.slider("Min Speaker Duration (s)", 0.5, 5.0, 1.0, 0.5)
    chunk_size = st.slider("Chunk Size (s)", 2, 10, 5, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px; background: #f0f2f6; border-radius: 5px;'>
    <small>🎧 Voice Separation System v2.0</small><br>
    <small>Powered by SepFormer + AssemblyAI</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN CONTENT
# ============================================

st.title("🎧 AI Voice Separation + Real-Time Transcription")
st.markdown("*Multi-speaker detection, separation, and live transcription with AssemblyAI*")

# ============================================
# MODE 1: UPLOAD AUDIO FILE
# ============================================

if input_mode == "Upload File":
    st.subheader("📤 Upload Audio File")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        audio_file = st.file_uploader("Choose audio file (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_btn = st.button("▶️ Process", use_container_width=True)
    
    if audio_file:
        try:
            audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)
            st.session_state.audio_loaded = True
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{len(audio_data)/sample_rate:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sample_rate}Hz")
            with col3:
                st.metric("File Size", f"{audio_file.size/1024:.1f}KB")
            
            st.success("✅ Audio loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# ============================================
# MODE 2: LIVE TRANSCRIPTION
# ============================================

elif input_mode == "🔴 Live Transcription":
    st.subheader("🔴 Live Microphone Input & Real-Time Transcription")
    
    if not api_key:
        st.warning("⚠️ Enter AssemblyAI API Key in sidebar to enable live transcription")
        st.info("**Getting started:**\n1. Sign up at https://www.assemblyai.com\n2. Get your free API key\n3. Paste it in the sidebar")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_live = st.button("🔴 Start Recording", key="start_live")
    with col2:
        stop_live = st.button("⏹️ Stop Recording", key="stop_live")
    with col3:
        download_live = st.button("📥 Download", key="download_live")
    
    status_container = st.container()
    transcript_container = st.container()
    stats_container = st.container()
    
    # Real-time transcription logic
    if start_live and not st.session_state.realtime_active:
        st.session_state.realtime_active = True
        st.session_state.full_transcript = []
        
        rtm = RealtimeTranscriptionManager(api_key)
        
        with status_container:
            st.markdown("""
                <div style='background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;'>
                <span class='status-live'></span> <b>Recording...</b> Speak into your microphone
                </div>
                """, unsafe_allow_html=True)
        
        if rtm.start_realtime_transcription():
            placeholder_transcript = st.empty()
            placeholder_stats = st.empty()
            
            try:
                while st.session_state.realtime_active:
                    updates = rtm.get_transcript_updates()
                    
                    for update in updates:
                        if 'text' in update and update['text']:
                            st.session_state.full_transcript.append(update['text'])
                        elif 'status' in update and update['status'] == 'completed':
                            st.session_state.realtime_active = False
                    
                    # Display live transcript
                    with placeholder_transcript.container():
                        st.markdown("### 📝 Live Transcript")
                        full_text = ' '.join(st.session_state.full_transcript)
                        st.markdown(f"""
                            <div class='transcript-box'>
                            {full_text if full_text else "Waiting for speech..."}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display stats
                    with placeholder_stats.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Segments", len(st.session_state.full_transcript))
                        with col2:
                            st.metric("Characters", len(full_text))
                        with col3:
                            st.metric("Words", len(full_text.split()))
                    
                    time.sleep(0.5)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            
            finally:
                rtm.save_session()
    
    if stop_live and st.session_state.realtime_active:
        st.session_state.realtime_active = False
        with status_container:
            st.markdown("""
                <div style='background: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;'>
                ✅ <b>Recording stopped</b> - Processing complete
                </div>
                """, unsafe_allow_html=True)
    
    # Final transcript display
    if st.session_state.full_transcript:
        st.markdown("---")
        st.subheader("✅ Final Transcript")
        final_text = ' '.join(st.session_state.full_transcript)
        
        st.markdown(f"""
            <div class='transcript-box'>
            {final_text}
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Words", len(final_text.split()))
        with col2:
            st.metric("Characters", len(final_text))
        with col3:
            st.metric("Segments", len(st.session_state.full_transcript))
        
        if download_live:
            transcript_json = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'transcript': final_text,
                'word_count': len(final_text.split()),
                'segment_count': len(st.session_state.full_transcript)
            }, indent=2)
            
            st.download_button(
                label="📥 Download as JSON",
                data=transcript_json,
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# ============================================
# MODE 3: SAMPLE AUDIO
# ============================================

elif input_mode == "Sample Audio":
    st.subheader("📻 Test with Sample Audio")
    st.info("Generate a sample audio with multiple speakers for testing")
    
    if st.button("📻 Generate Sample Audio"):
        with st.spinner("Generating sample audio..."):
            # Create synthetic multi-speaker audio
            sample_rate = 16000
            duration = 10
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Speaker 1: Lower frequency
            speaker1 = np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 0.1 * t)
            
            # Speaker 2: Higher frequency
            speaker2 = np.sin(2 * np.pi * 300 * t) * np.sin(2 * np.pi * 0.15 * t)
            
            # Mix with random noise
            mixed = (speaker1 + speaker2) / 2 + np.random.normal(0, 0.05, len(t))
            mixed = mixed / np.max(np.abs(mixed))
            
            st.session_state.audio_data = mixed
            st.session_state.sample_rate = sample_rate
            st.session_state.audio_loaded = True
            
            st.audio(mixed, sample_rate=sample_rate, format="audio/wav")
            st.success("✅ Sample audio generated")

# ============================================
# AUDIO PROCESSING
# ============================================

if st.session_state.audio_loaded and st.session_state.audio_data is not None:
    if input_mode != "🔴 Live Transcription":  # Skip for live mode
        
        audio_data = st.session_state.audio_data
        sample_rate = st.session_state.sample_rate
        
        if st.button("▶️ START PROCESSING", use_container_width=True, key="process_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Visualization
                st.subheader("📊 Audio Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Waveform")
                    fig_wave = create_visualizations.plot_waveform(audio_data, sample_rate)
                    st.plotly_chart(fig_wave, use_container_width=True)
                
                with col2:
                    st.markdown("#### Spectrogram")
                    fig_spec = create_visualizations.plot_spectrogram(audio_data, sample_rate)
                    st.plotly_chart(fig_spec, use_container_width=True)
                
                progress_bar.progress(15)
                status_text.info("📊 Analysis complete...")
                
                # Step 2: Speaker Detection
                st.subheader("👥 Speaker Detection")
                
                diarization = SpeakerDiarization(model_name=diarization_model)
                num_speakers, _ = diarization.detect_speakers(
                    audio_data, sample_rate, min_duration=min_duration
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎤 Speakers Detected", num_speakers)
                with col2:
                    st.metric("⏱️ Duration", f"{len(audio_data)/sample_rate:.2f}s")
                with col3:
                    st.metric("📍 Sample Rate", f"{sample_rate}Hz")
                
                progress_bar.progress(35)
                status_text.info(f"👥 Detected {num_speakers} speakers...")
                
                # Step 3: Speech Separation
                st.subheader("🔊 Speech Separation")
                
                separator = SpeechSeparation(model_name=separation_model)
                separated_audio = separator.separate(
                    audio_data, sample_rate, num_speakers=num_speakers
                )
                
                progress_bar.progress(55)
                status_text.info("🔊 Separating speakers...")
                
                # Step 4: Noise Reduction
                st.subheader("🧹 Noise Reduction & Enhancement")
                
                noise_reducer = NoiseReduction()
                enhanced_audio = []
                
                for i, speaker_audio in enumerate(separated_audio):
                    cleaned = noise_reducer.reduce_noise(
                        speaker_audio, sample_rate, strength=noise_strength
                    )
                    if enhance_voice:
                        cleaned = noise_reducer.enhance_voice(cleaned, sample_rate)
                    enhanced_audio.append(cleaned)
                
                progress_bar.progress(70)
                status_text.info("🧹 Enhancing audio...")
                
                # Step 5: Quality Metrics
                st.subheader("📈 Audio Quality Metrics")
                
                metrics_data = []
                for i, speaker_audio in enumerate(enhanced_audio):
                    snr, sdr = compute_metrics(separated_audio[i], speaker_audio)
                    metrics_data.append({
                        "Speaker": f"Speaker {i+1}",
                        "SNR (dB)": f"{snr:.2f}",
                        "SDR (dB)": f"{sdr:.2f}"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                progress_bar.progress(80)
                status_text.info("📈 Computing metrics...")
                
                # Step 6: Transcription
                st.subheader("🗣️ Speech-to-Text Transcription")
                
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
                
                progress_bar.progress(85)
                
                # Step 7: Output Speakers
                st.subheader("🎧 Separated Speaker Outputs")
                
                for i, (speaker_audio, transcript) in enumerate(zip(enhanced_audio, transcriptions)):
                    with st.container():
                        st.markdown(f"<div class='speaker-section'>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"### 🎤 Speaker {i+1}")
                        with col2:
                            st.markdown(f"**{transcript['language']}**")
                        with col3:
                            is_silent = np.mean(np.abs(speaker_audio)) < 0.01
                            st.markdown("⚪ Silent" if is_silent else "🟢 Active")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            snr_val = float(metrics_df[metrics_df['Speaker'] == f"Speaker {i+1}"]['SNR (dB)'].values[0])
                            st.metric("SNR", f"{snr_val:.2f}dB")
                        with col2:
                            sdr_val = float(metrics_df[metrics_df['Speaker'] == f"Speaker {i+1}"]['SDR (dB)'].values[0])
                            st.metric("SDR", f"{sdr_val:.2f}dB")
                        
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
                        
                        # Download button
                        output_path = Path("outputs") / f"speaker_{i+1}.wav"
                        output_path.parent.mkdir(exist_ok=True)
                        sf.write(str(output_path), speaker_audio, sample_rate)
                        
                        with open(str(output_path), "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download Speaker {i+1} Audio",
                                data=f.read(),
                                file_name=f"speaker_{i+1}.wav",
                                mime="audio/wav",
                                key=f"download_{i}"
                            )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                progress_bar.progress(95)
                
                # Step 8: Save Session
                st.subheader("📋 Session Report")
                
                report_data = {
                    "Timestamp": datetime.now().isoformat(),
                    "Duration (s)": f"{len(audio_data)/sample_rate:.2f}",
                    "Speakers": num_speakers,
                    "Sample Rate": sample_rate,
                    "Models": f"{separation_model} + {diarization_model}"
                }
                
                report_df = pd.DataFrame([report_data])
                st.dataframe(report_df, use_container_width=True)
                
                # Save logs
                logs_path = Path("logs")
                logs_path.mkdir(exist_ok=True)
                
                report_csv = logs_path / "results.csv"
                if report_csv.exists():
                    existing = pd.read_csv(report_csv)
                    report_df = pd.concat([existing, report_df], ignore_index=True)
                
                report_df.to_csv(str(report_csv), index=False)
                
                csv_data = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Report",
                    data=csv_data,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                progress_bar.progress(100)
                status_text.success("✅ Processing complete!")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
    <p><b>🎧 AI Voice Separation System v2.0</b></p>
    <p>SepFormer + ECAPA-TDNN + AssemblyAI Real-Time</p>
    <p><small>No FFmpeg Required • Production Ready</small></p>
    </div>
    """, unsafe_allow_html=True)