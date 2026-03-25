# 🎧 AI Voice Separation + Real-Time Transcription System

Production-ready deployment of multi-speaker voice separation with AssemblyAI real-time transcription.

## ✨ Features

- 🎙️ **Live Microphone Input** - Real-time audio capture
- 🔊 **Multi-Speaker Detection** - Automatically detect 2+ speakers
- 🎵 **Audio Separation** - Individual clean audio streams per speaker
- 🗣️ **Real-Time Transcription** - AssemblyAI integration
- 🧹 **Noise Reduction** - Automatic noise removal
- 📊 **Quality Metrics** - SNR & SDR calculations
- ⬇️ **Download Support** - Export audio & transcripts

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- Microphone
- AssemblyAI API key (free)

### Installation

```bash
# Clone repo
git clone https://github.com/archanaj1412/speech-separation-system.git
cd speech-separation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app_assemblyai.py