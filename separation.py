import numpy as np
import torch
import torchaudio
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SpeechSeparation:
    """
    Speech separation using deep learning models.
    Supports SepFormer, Conv-TasNet, and Demucs.
    """
    
    def __init__(self, model_name: str = "SepFormer"):
        self.model_name = model_name
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self):
        """Load pretrained separation model"""
        try:
            if self.model_name == "SepFormer":
                return self._load_sepformer()
            elif self.model_name == "Conv-TasNet":
                return self._load_convtasnet()
            elif self.model_name == "Demucs":
                return self._load_demucs()
        except Exception as e:
            print(f"Warning: Could not load {self.model_name}: {e}")
            return None
    
    def _load_sepformer(self):
        """Load SepFormer model from SpeechBrain"""
        try:
            from speechbrain.pretrained import SepformerSeparation
            return SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir="pretrained_models/sepformer-wsj02mix",
                run_opts={"device": self.device}
            )
        except Exception as e:
            print(f"Could not load SepFormer: {e}")
            return None
    
    def _load_convtasnet(self):
        """Load Conv-TasNet model"""
        try:
            from speechbrain.pretrained import ConvTasNet
            model = ConvTasNet.from_hparams(
                source="speechbrain/conv-tasnet-wsj02mix",
                savedir="pretrained_models/conv-tasnet-wsj02mix",
                run_opts={"device": self.device}
            )
            return model
        except Exception as e:
            print(f"Could not load Conv-TasNet: {e}")
            return None
    
    def _load_demucs(self):
        """Load Demucs model"""
        try:
            import demucs.pretrained
            return demucs.pretrained.get_model('mdx')
        except Exception as e:
            print(f"Could not load Demucs: {e}")
            return None
    
    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int = 2
    ) -> List[np.ndarray]:
        """
        Separate speakers from mixed audio.
        
        Args:
            audio: Mixed audio signal
            sample_rate: Sample rate
            num_speakers: Number of speakers to separate
        
        Returns:
            List of separated speaker audio
        """
        
        if self.model is None:
            return self._fallback_separation(audio, num_speakers)
        
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_tensor = resampler(audio_tensor)
                sample_rate = 16000
            
            audio_tensor = audio_tensor.to(self.device)
            
            # Separate using appropriate model
            if self.model_name == "SepFormer" or self.model_name == "Conv-TasNet":
                separated = self.model.separate_batch(audio_tensor)
            elif self.model_name == "Demucs":
                separated = self._separate_demucs(audio_tensor, num_speakers)
            else:
                separated = self._fallback_separation(audio, num_speakers)
            
            # Convert output to numpy
            if isinstance(separated, torch.Tensor):
                separated = separated.cpu().numpy()
            
            # Extract individual speakers
            speaker_audios = []
            for i in range(min(num_speakers, separated.shape[0])):
                speaker_audio = separated[i, 0, :] if separated.ndim == 3 else separated[i, :]
                # Resample back to original sample rate if needed
                if sample_rate != 16000:
                    speaker_audio = self._resample(speaker_audio, 16000, sample_rate)
                speaker_audios.append(speaker_audio)
            
            return speaker_audios
        
        except Exception as e:
            print(f"Warning: Separation failed: {e}")
            return self._fallback_separation(audio, num_speakers)
    
    def _separate_demucs(self, audio: torch.Tensor, num_speakers: int) -> torch.Tensor:
        """Separate using Demucs model"""
        with torch.no_grad():
            separated = self.model.separate(audio)
        return separated[:num_speakers]
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio"""
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def _fallback_separation(self, audio: np.ndarray, num_speakers: int) -> List[np.ndarray]:
        """
        Fallback separation using simple spectral masking.
        """
        # Compute STFT
        D = self._compute_stft(audio)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Simple clustering-based mask
        masks = self._create_masks(magnitude, num_speakers)
        
        # Apply masks and inverse STFT
        separated = []
        for mask in masks:
            masked = magnitude * mask
            X = masked * np.exp(1j * phase)
            x = self._inverse_stft(X)
            separated.append(x)
        
        return separated
    
    def _compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute STFT"""
        return np.fft.rfft(audio)
    
    def _inverse_stft(self, X: np.ndarray) -> np.ndarray:
        """Inverse STFT"""
        return np.fft.irfft(X)
    
    def _create_masks(self, magnitude: np.ndarray, num_speakers: int) -> List[np.ndarray]:
        """Create spectral masks for speakers"""
        masks = []
        threshold = np.mean(magnitude)
        
        for i in range(num_speakers):
            mask = (magnitude > (threshold * (1 - i * 0.2))).astype(float)
            masks.append(mask)
        
        return masks