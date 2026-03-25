import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class SpeechToText:
    """
    Speech-to-text transcription using Whisper AI.
    Also supports language detection.
    """
    
    def __init__(self):
        self.model = self._load_whisper()
    
    def _load_whisper(self):
        """Load Whisper model"""
        try:
            import openai
            # Note: Requires OpenAI API key or local Whisper installation
            # Using local version:
            import whisper
            return whisper.load_model("base")
        except Exception as e:
            print(f"Warning: Could not load Whisper model: {e}")
            return None
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
        
        Returns:
            Transcribed text
        """
        
        if self.model is None:
            return self._mock_transcription(audio)
        
        try:
            import whisper
            import tempfile
            import soundfile as sf
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sample_rate)
                temp_path = f.name
            
            # Transcribe
            result = self.model.transcribe(temp_path, language="en")
            
            # Clean up
            import os
            os.unlink(temp_path)
            
            return result.get("text", "")
        
        except Exception as e:
            print(f"Warning: Transcription failed: {e}")
            return self._mock_transcription(audio)
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of transcribed text.
        
        Args:
            text: Transcribed text
        
        Returns:
            Detected language code
        """
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            detected_lang = blob.detect_language()
            return self._lang_code_to_name(detected_lang)
        except Exception as e:
            print(f"Warning: Language detection failed: {e}")
            return "Unknown"
    
    def _lang_code_to_name(self, code: str) -> str:
        """Convert language code to name"""
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'hi': 'Hindi',
            'ar': 'Arabic',
        }
        return language_map.get(code, code.upper())
    
    def _mock_transcription(self, audio: np.ndarray) -> str:
        """Return mock transcription for fallback"""
        duration = len(audio) / 16000
        return f"[Mock Transcription] Audio duration: {duration:.2f} seconds"
    
    def extract_keywords(self, text: str) -> list:
        """
        Extract keywords from transcribed text.
        
        Args:
            text: Transcribed text
        
        Returns:
            List of keywords
        """
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            import nltk
            
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            keywords = [t for t in tokens if t.isalnum() and t not in stop_words]
            
            return keywords[:10]  # Top 10 keywords
        except Exception as e:
            print(f"Warning: Keyword extraction failed: {e}")
            return []
    
    def detect_emotion(self, text: str) -> dict:
        """
        Simple emotion detection from text.
        
        Args:
            text: Transcribed text
        
        Returns:
            Emotion scores
        """
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.5:
                emotion = "Very Positive 😄"
            elif polarity > 0.1:
                emotion = "Positive 🙂"
            elif polarity < -0.5:
                emotion = "Very Negative 😢"
            elif polarity < -0.1:
                emotion = "Negative 😟"
            else:
                emotion = "Neutral 😐"
            
            return {
                "emotion": emotion,
                "polarity": polarity,
                "confidence": abs(polarity)
            }
        except Exception as e:
            print(f"Warning: Emotion detection failed: {e}")
            return {"emotion": "Unknown", "polarity": 0, "confidence": 0}