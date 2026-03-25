"""
AssemblyAI Real-Time Transcription Integration
Handles live audio streaming and real-time transcription
"""

import assemblyai as aai
import asyncio
import threading
import queue
from typing import Callable, Optional, Dict, List
import numpy as np
import streamlit as st
from datetime import datetime
import json

class RealtimeTranscriptionManager:
    """Manages real-time transcription with AssemblyAI"""
    
    def __init__(self, api_key: str):
        """Initialize with AssemblyAI API key"""
        aai.settings.api_key = api_key
        self.transcriber = None
        self.is_recording = False
        self.transcript_queue = queue.Queue()
        self.full_transcript = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def on_data(self, transcript):
        """Callback when transcript data is received"""
        if transcript.text:
            self.transcript_queue.put({
                'text': transcript.text,
                'confidence': transcript.confidence,
                'timestamp': datetime.now().isoformat(),
                'is_final': not transcript.partial
            })
            
            if not transcript.partial:
                self.full_transcript.append(transcript.text)
    
    def on_error(self, error):
        """Callback for transcription errors"""
        self.transcript_queue.put({
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
    
    def on_end(self):
        """Callback when transcription ends"""
        self.transcript_queue.put({
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        })
    
    def start_realtime_transcription(self):
        """Start real-time transcription from microphone"""
        try:
            self.transcriber = aai.RealtimeTranscriber(
                on_data=self.on_data,
                on_error=self.on_error,
                on_end=self.on_end,
                encoding=aai.AudioEncoding.pcm_16,
                sample_rate=16000,
            )
            
            self.transcriber.connect()
            self.is_recording = True
            
            # Start audio capture in background thread
            audio_thread = threading.Thread(target=self._capture_audio, daemon=True)
            audio_thread.start()
            
            return True
        except Exception as e:
            st.error(f"❌ Failed to start transcription: {str(e)}")
            return False
    
    def _capture_audio(self):
        """Capture audio from microphone and send to AssemblyAI"""
        try:
            import pyaudio
            
            CHUNK = 4096
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            while self.is_recording and self.transcriber and self.transcriber.is_connected():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    self.transcriber.send(data)
                except Exception as e:
                    print(f"Audio capture error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except ImportError:
            st.error("❌ PyAudio not installed. Install with: pip install pyaudio")
        except Exception as e:
            st.error(f"❌ Audio capture error: {str(e)}")
    
    def stop_transcription(self):
        """Stop real-time transcription"""
        if self.transcriber:
            self.is_recording = False
            self.transcriber.close()
            return True
        return False
    
    def get_transcript_updates(self) -> List[Dict]:
        """Get all pending transcript updates"""
        updates = []
        while not self.transcript_queue.empty():
            try:
                updates.append(self.transcript_queue.get_nowait())
            except queue.Empty:
                break
        return updates
    
    def get_full_transcript(self) -> str:
        """Get complete transcript"""
        return ' '.join(self.full_transcript)
    
    def save_session(self, output_path: str = "logs/transcription_sessions.json"):
        """Save transcription session"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        session_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'transcript': self.get_full_transcript(),
            'num_segments': len(self.full_transcript)
        }
        
        # Append to sessions file
        try:
            with open(output_path, 'r') as f:
                sessions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            sessions = []
        
        sessions.append(session_data)
        
        with open(output_path, 'w') as f:
            json.dump(sessions, f, indent=2)
        
        return session_data


class MultiSpeakerRealtimeAnalyzer:
    """Real-time multi-speaker separation with transcription"""
    
    def __init__(self, api_key: str, diarization_model: str = "ECAPA-TDNN"):
        from diarization import SpeakerDiarization
        from separation import SpeechSeparation
        from transcription import SpeechToText
        from noise_reduction import NoiseReduction
        
        self.rtm = RealtimeTranscriptionManager(api_key)
        self.diarization = SpeakerDiarization(model_name=diarization_model)
        self.separation = SpeechSeparation()
        self.transcription = SpeechToText()
        self.noise_reducer = NoiseReduction()
        
        self.speaker_data = {}  # Store per-speaker data
    
    def process_realtime_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000):
        """Process audio chunk in real-time"""
        try:
            # Detect speakers in chunk
            num_speakers, _ = self.diarization.detect_speakers(
                audio_chunk, sample_rate, min_duration=0.5
            )
            
            # Separate speakers
            separated = self.separation.separate(
                audio_chunk, sample_rate, num_speakers=num_speakers
            )
            
            # Enhance and transcribe each speaker
            for i, speaker_audio in enumerate(separated):
                # Noise reduction
                enhanced = self.noise_reducer.reduce_noise(
                    speaker_audio, sample_rate, strength=0.7
                )
                
                # Transcribe
                text = self.transcription.transcribe(enhanced, sample_rate)
                language = self.transcription.detect_language(text)
                
                # Store speaker data
                speaker_key = f"Speaker_{i+1}"
                if speaker_key not in self.speaker_data:
                    self.speaker_data[speaker_key] = {
                        'text': [],
                        'language': language,
                        'audio_chunks': []
                    }
                
                self.speaker_data[speaker_key]['text'].append(text)
                self.speaker_data[speaker_key]['audio_chunks'].append(enhanced)
            
            return num_speakers, self.speaker_data
        
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return None, None
    
    def get_speaker_summary(self) -> Dict:
        """Get summary of all speakers"""
        summary = {}
        for speaker, data in self.speaker_data.items():
            summary[speaker] = {
                'full_text': ' '.join(data['text']),
                'language': data['language'],
                'num_chunks': len(data['text']),
                'total_duration': sum([len(chunk) / 16000 for chunk in data['audio_chunks']])
            }
        return summary