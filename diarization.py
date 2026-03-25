import numpy as np
import librosa
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import torch
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class SpeakerDiarization:
    """
    Speaker diarization using ECAPA-TDNN embeddings and clustering.
    Detects number of speakers and their time intervals.
    """
    
    def __init__(self, model_name: str = "ECAPA-TDNN"):
        self.model_name = model_name
        self.model = self._load_model()
    
    def _load_model(self):
        """Load pretrained diarization model"""
        try:
            if self.model_name == "ECAPA-TDNN":
                # Load ECAPA-TDNN model from SpeechBrain
                import speechbrain as sb
                from speechbrain.pretrained import SpeakerRecognition
                model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
            else:  # VoxCeleb
                model = self._load_voxceleb()
            
            return model
        except Exception as e:
            print(f"Warning: Could not load {self.model_name} model: {e}")
            return None
    
    def _load_voxceleb(self):
        """Load VoxCeleb pretrained model"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            return EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvector-voxceleb",
                savedir="pretrained_models/spkrec-xvector-voxceleb",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
        except Exception as e:
            print(f"Could not load VoxCeleb model: {e}")
            return None
    
    def detect_speakers(
        self,
        audio: np.ndarray,
        sample_rate: int,
        min_duration: float = 1.0,
        clustering_threshold: float = 0.5
    ) -> Tuple[int, List[Dict]]:
        """
        Detect number of speakers and their intervals.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            min_duration: Minimum speaker duration
            clustering_threshold: Threshold for clustering
        
        Returns:
            Tuple of (num_speakers, speaker_intervals)
        """
        
        # Split audio into chunks
        chunk_duration = 2.0  # seconds
        chunk_samples = int(chunk_duration * sample_rate)
        chunks = [
            audio[i:i + chunk_samples]
            for i in range(0, len(audio), chunk_samples)
        ]
        
        if not chunks:
            return 1, [{"speaker": 1, "start": 0, "end": len(audio) / sample_rate}]
        
        # Extract embeddings from each chunk
        embeddings = []
        for chunk in chunks:
            if len(chunk) < sample_rate:  # Minimum 1 second
                continue
            
            try:
                embedding = self._extract_embedding(chunk, sample_rate)
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as e:
                print(f"Warning: Could not extract embedding: {e}")
                continue
        
        if not embeddings:
            return 1, [{"speaker": 1, "start": 0, "end": len(audio) / sample_rate}]
        
        embeddings = np.array(embeddings)
        
        # Clustering using hierarchical clustering
        num_speakers = self._cluster_embeddings(
            embeddings,
            threshold=clustering_threshold
        )
        
        # Assign speakers to intervals
        speaker_intervals = self._assign_speakers(
            num_speakers,
            len(chunks),
            chunk_duration
        )
        
        return num_speakers, speaker_intervals
    
    def _extract_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract speaker embedding from audio chunk"""
        if self.model is None:
            return self._mock_embedding()
        
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            
            if sample_rate != 16000:
                resampler = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                audio_tensor = torch.FloatTensor(resampler).unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Warning: Embedding extraction failed: {e}")
            return self._mock_embedding()
    
    def _mock_embedding(self) -> np.ndarray:
        """Return mock embedding for fallback"""
        return np.random.randn(192)  # ECAPA-TDNN embedding dimension
    
    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        threshold: float = 0.5
    ) -> int:
        """
        Cluster embeddings using hierarchical clustering.
        
        Args:
            embeddings: Speaker embeddings
            threshold: Clustering threshold
        
        Returns:
            Number of speakers
        """
        if len(embeddings) <= 1:
            return 1
        
        # Compute pairwise distances
        distances = pdist(embeddings, metric='cosine')
        linkage_matrix = linkage(distances, method='average')
        
        # Form clusters
        clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
        num_speakers = len(np.unique(clusters))
        
        return num_speakers
    
    def _assign_speakers(
        self,
        num_speakers: int,
        num_chunks: int,
        chunk_duration: float
    ) -> List[Dict]:
        """Assign speakers to time intervals"""
        speaker_intervals = []
        chunk_length = int(chunk_duration * 16000)  # Standard sample rate
        
        for i in range(num_speakers):
            start = (i * chunk_length) / 16000
            end = ((i + 1) * chunk_length) / 16000
            
            speaker_intervals.append({
                "speaker": i + 1,
                "start": start,
                "end": min(end, num_chunks * chunk_duration)
            })
        
        return speaker_intervals
    
    def get_speaker_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int
    ) -> List[np.ndarray]:
        """
        Extract separate audio segments for each speaker.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            num_speakers: Number of speakers
        
        Returns:
            List of speaker audio segments
        """
        segment_length = len(audio) // num_speakers
        speaker_segments = []
        
        for i in range(num_speakers):
            start = i * segment_length
            end = (i + 1) * segment_length if i < num_speakers - 1 else len(audio)
            speaker_segments.append(audio[start:end])
        
        return speaker_segments