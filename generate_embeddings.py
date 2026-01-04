"""
Generate speaker embeddings using SpeechBrain ECAPA-TDNN model.
Official implementation following SpeechBrain documentation.
"""

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
import warnings
warnings.filterwarnings('ignore')


class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN model."""
    
    def __init__(self, device=None):
        """
        Initialize the embedding extractor.
        
        Args:
            device: torch device (default: auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading SpeechBrain ECAPA-TDNN model on {self.device}...")
        
        # Load pretrained model using official SpeechBrain API
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY  
)

        print("✓ Model loaded successfully")
        print("  Model: ECAPA-TDNN")

    
    def extract_embedding(self, audio_segment, sample_rate=16000):
        """
        Extract embedding from a single audio segment.
        
        Args:
            audio_segment: numpy array of audio samples (int16 or float)
            sample_rate: sample rate of audio (default: 16000)
        
        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        # Convert to float if needed
        if audio_segment.dtype == np.int16:
            audio_float = audio_segment.astype(np.float32) / 32768.0
        else:
            audio_float = audio_segment.astype(np.float32)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_float).float()
        
        # Add batch dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Extract embedding using SpeechBrain's encode_batch
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(audio_tensor)
            embedding = embeddings.squeeze().cpu().numpy()
        
        return embedding
    
    def extract_embeddings_batch(self, segments, sample_rate=16000):
        """
        Extract embeddings from multiple segments.
        
        Args:
            segments: List of Segment namedtuples with audio_data
            sample_rate: sample rate of audio (must be 16000 for this model)
        
        Returns:
            embeddings: numpy array of shape (num_segments, embedding_dim)
        """
        if sample_rate != 16000:
            print(f"⚠ Warning: Model expects 16kHz audio, got {sample_rate}Hz")
            print("  Results may be suboptimal. Consider resampling to 16kHz.")
        
        print(f"Extracting embeddings from {len(segments)} segments...")
        
        embeddings = []
        
        for i, segment in enumerate(segments):
            if (i + 1) % 50 == 0 or (i + 1) == len(segments):
                print(f"  Progress: {i + 1}/{len(segments)}")
            
            try:
                embedding = self.extract_embedding(segment.audio_data, sample_rate)
                embeddings.append(embedding)
            except Exception as e:
                print(f"  ⚠ Warning: Failed to extract embedding for segment {i}: {e}")
                # Add zero embedding as placeholder
                if len(embeddings) > 0:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    embeddings.append(np.zeros(192))  # ECAPA-TDNN embedding size
        
        embeddings = np.array(embeddings)
        
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        return embeddings


def generate_embeddings_from_segments(segments, sample_rate=16000):
    """
    Convenience function to generate embeddings from segments.
    
    Args:
        segments: List of audio segments (Segment namedtuples)
        sample_rate: Audio sample rate
    
    Returns:
        embeddings: numpy array of embeddings
    """
    extractor = SpeakerEmbeddingExtractor()
    embeddings = extractor.extract_embeddings_batch(segments, sample_rate)

    print(f"\nFinal embedding matrix shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    return embeddings

    
