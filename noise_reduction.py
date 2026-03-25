import numpy as np
import librosa
from scipy import signal
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class NoiseReduction:
    """
    Noise reduction and voice enhancement using spectral gating
    and pretrained denoisers.
    """
    
    def __init__(self):
        self.denoiser = self._load_denoiser()
    
    def _load_denoiser(self):
        """Load pretrained denoising model"""
        try:
            import torch
            from speechbrain.pretrained import SpectralMaskEnhancement
            denoiser = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/enhance_metricgan_plus-DNS",
                savedir="pretrained_models/enhance_metricgan_plus",
            )
            return denoiser
        except Exception as e:
            print(f"Warning: Could not load denoiser: {e}")
            return None
    
    def reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float = 0.7
    ) -> np.ndarray:
        """
        Reduce noise from audio using spectral gating.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            strength: Noise reduction strength (0-1)
        
        Returns:
            Denoised audio
        """
        
        # Method 1: Spectral gating
        denoised = self._spectral_gating(audio, sample_rate, strength)
        
        # Method 2: Apply pretrained denoiser if available
        if self.denoiser is not None:
            try:
                denoised = self._denoise_with_model(denoised, sample_rate)
            except Exception as e:
                print(f"Denoiser model failed, using spectral gating: {e}")
        
        return denoised
    
    def _spectral_gating(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float
    ) -> np.ndarray:
        """
        Apply spectral gating for noise reduction.
        """
        # Compute STFT
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Estimate noise profile from quiet regions
        noise_profile = self._estimate_noise_profile(magnitude)
        
        # Create spectral gate
        gate = np.maximum(magnitude - strength * noise_profile, 0) / (magnitude + 1e-10)
        
        # Apply gate
        D_gated = gate * D
        
        # Inverse STFT
        audio_denoised = librosa.istft(D_gated)
        
        return audio_denoised
    
    def _estimate_noise_profile(self, magnitude: np.ndarray) -> np.ndarray:
        """Estimate noise profile from quiet regions"""
        # Use percentile as noise floor
        noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
        return noise_floor
    
    def _denoise_with_model(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply pretrained denoising model"""
        import torch
        
        try:
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            
            with torch.no_grad():
                denoised = self.denoiser(audio_tensor)
            
            return denoised.squeeze().numpy()
        except Exception as e:
            print(f"Model denoising failed: {e}")
            return audio
    
    def enhance_voice(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhance voice clarity using various techniques.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
        
        Returns:
            Enhanced audio
        """
        # Step 1: High-pass filter (remove rumble)
        audio = self._highpass_filter(audio, sample_rate, cutoff=80)
        
        # Step 2: Loudness normalization
        audio = self._normalize_loudness(audio)
        
        # Step 3: Gentle compression
        audio = self._apply_compression(audio, ratio=2.0, threshold=-30)
        
        return audio
    
    def _highpass_filter(
        self,
        audio: np.ndarray,
        sample_rate: int,
        cutoff: float = 80
    ) -> np.ndarray:
        """Apply high-pass filter"""
        sos = signal.butter(4, cutoff, 'hp', fs=sample_rate, output='sos')
        return signal.sosfilt(sos, audio)
    
    def _normalize_loudness(self, audio: np.ndarray, target_loudness: float = -20.0) -> np.ndarray:
        """Normalize loudness using LUFS approximation"""
        # Simple RMS-based normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_loudness / 20)
            audio = audio * (target_rms / rms)
        
        # Soft clipping to prevent distortion
        return np.tanh(audio)
    
    def _apply_compression(
        self,
        audio: np.ndarray,
        ratio: float = 2.0,
        threshold: float = -30,
        attack_time: float = 0.005,
        release_time: float = 0.1
    ) -> np.ndarray:
        """Apply dynamic range compression"""
        # Convert threshold from dB
        threshold_linear = 10 ** (threshold / 20)
        
        # Detect envelope
        envelope = np.abs(audio)
        
        # Apply compression
        mask = envelope > threshold_linear
        compressed = audio.copy()
        compressed[mask] = audio[mask] * (threshold_linear / (envelope[mask] ** (1 - 1/ratio)))
        
        return compressed
    
    def remove_echo(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Simple echo removal using cepstral analysis.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
        
        Returns:
            Echo-removed audio
        """
        # Compute cepstrum
        spectrum = np.abs(np.fft.rfft(audio))
        cepstrum = np.fft.irfft(np.log(spectrum + 1e-10))
        
        # Zero out echo delays (high quefrency)
        max_echo_delay = int(0.1 * sample_rate)  # 100ms max echo
        cepstrum[max_echo_delay:] = 0
        
        # Inverse cepstrum
        spectrum_new = np.exp(np.fft.rfft(cepstrum))
        audio_output = np.fft.irfft(spectrum_new, n=len(audio))
        
        return audio_output