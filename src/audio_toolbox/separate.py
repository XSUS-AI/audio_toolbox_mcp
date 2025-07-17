import numpy as np
import torch
from demucs.pretrained import get_model as get_demucs_model
from demucs.apply import apply_model
from typing import Optional, Tuple, Union, Dict, Any, List, TYPE_CHECKING
from pydantic import BaseModel, Field
from pathlib import Path
import os

# Type checking imports
if TYPE_CHECKING:
    from audio_toolbox.processor import AudioProcessor

class SeparationRequest(BaseModel):
    """Parameters for audio source separation"""
    model_name: str = Field("htdemucs", description="Model name to use for separation (htdemucs, htdemucs_ft, mdx, etc.)")
    shifts: int = Field(0, description="Number of random shifts for prediction averaging")
    overlap: float = Field(0.25, description="Overlap between prediction windows")
    segments: Optional[int] = Field(None, description="Length of segments for processing in seconds")
    split: bool = Field(True, description="Split audio into chunks for GPU memory efficiency")
    device: Optional[str] = Field(None, description="Device to use for processing (cpu, cuda)")

class TwoStemSeparationRequest(BaseModel):
    """Parameters for separating audio into two stems"""
    model_name: str = Field("htdemucs", description="Model name to use for separation (htdemucs, htdemucs_ft, mdx, etc.)")
    stem: str = Field("vocals", description="Stem to isolate (vocals, drums, bass, other, etc.)")
    shifts: int = Field(0, description="Number of random shifts for prediction averaging")
    overlap: float = Field(0.25, description="Overlap between prediction windows")
    segments: Optional[int] = Field(None, description="Length of segments for processing in seconds")
    split: bool = Field(True, description="Split audio into chunks for GPU memory efficiency")
    device: Optional[str] = Field(None, description="Device to use for processing (cpu, cuda)")

class AudioSeparator:
    """
    Audio source separation functionality using Demucs.
    
    This class provides methods for separating audio into different stems.
    """
    
    def __init__(self, processor: 'AudioProcessor'):
        """
        Initialize the separator with a reference to the AudioProcessor.
        
        Args:
            processor: The parent AudioProcessor instance
        """
        self._processor = processor
        self._model_cache = {}
    
    def _check_audio(self):
        """Check that audio data is available for separation"""
        if self._processor.audio_data is None or self._processor.sample_rate is None:
            raise ValueError("No audio data available for separation")
    
    def _get_model(self, model_name: str):
        """Get or load a Demucs model"""
        if model_name not in self._model_cache:
            self._model_cache[model_name] = get_demucs_model(model_name)
        return self._model_cache[model_name]
    
    def _prepare_audio_for_demucs(self):
        """Prepare audio in the format expected by Demucs"""
        # Demucs expects audio in shape (batch, channels, time)
        # Convert from our (channels, time) format
        return torch.tensor(self._processor.audio_data).unsqueeze(0)
    
    def separate_stems(self, request: Optional[SeparationRequest] = None) -> Dict[str, np.ndarray]:
        """
        Separate audio into multiple stems.
        
        Args:
            request: Parameters for audio separation
            
        Returns:
            Dictionary of separated stems (vocals, drums, bass, other, etc.)
        """
        self._check_audio()
        
        if request is None:
            request = SeparationRequest()
        
        # Get the model
        model = self._get_model(request.model_name)
        
        # Prepare audio
        audio = self._prepare_audio_for_demucs()
        
        # Determine device
        device = request.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move to device
        model.to(device)
        audio = audio.to(device)
        
        # Apply the model
        with torch.no_grad():
            sources = model(audio)  # sources is (batch, source, channels, time)
        
        # Convert to numpy and create result dict
        result = {}
        for source_idx, source_name in enumerate(model.sources):
            source_audio = sources[0, source_idx].cpu().numpy()  # (channels, time)
            result[source_name] = source_audio
        
        return result
    
    def vocals(self, request: Optional[TwoStemSeparationRequest] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate vocals from audio.
        
        Args:
            request: Parameters for vocal separation
            
        Returns:
            Tuple containing (vocals, accompaniment)
        """
        if request is None:
            request = TwoStemSeparationRequest(stem="vocals")
        else:
            request.stem = "vocals"
        
        return self.two_stems(request)
    
    def drums(self, request: Optional[TwoStemSeparationRequest] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate drums from audio.
        
        Args:
            request: Parameters for drum separation
            
        Returns:
            Tuple containing (drums, accompaniment without drums)
        """
        if request is None:
            request = TwoStemSeparationRequest(stem="drums")
        else:
            request.stem = "drums"
        
        return self.two_stems(request)
    
    def bass(self, request: Optional[TwoStemSeparationRequest] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate bass from audio.
        
        Args:
            request: Parameters for bass separation
            
        Returns:
            Tuple containing (bass, accompaniment without bass)
        """
        if request is None:
            request = TwoStemSeparationRequest(stem="bass")
        else:
            request.stem = "bass"
        
        return self.two_stems(request)
    
    def two_stems(self, request: Optional[TwoStemSeparationRequest] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate audio into two stems: target stem and everything else.
        
        Args:
            request: Parameters for two-stem separation
            
        Returns:
            Tuple containing (target_stem, everything_else)
        """
        self._check_audio()
        
        if request is None:
            request = TwoStemSeparationRequest()
            
        # Get all stems
        stems = self.separate_stems(SeparationRequest(
            model_name=request.model_name,
            shifts=request.shifts,
            overlap=request.overlap,
            segments=request.segments,
            split=request.split,
            device=request.device
        ))
        
        if request.stem not in stems:
            available_stems = list(stems.keys())
            raise ValueError(f"Requested stem '{request.stem}' not available. Available stems are: {available_stems}")
        
        # Get target stem
        target_stem = stems[request.stem]
        
        # Create the accompaniment by summing all other stems
        everything_else = np.zeros_like(target_stem)
        for stem_name, stem_audio in stems.items():
            if stem_name != request.stem:
                everything_else += stem_audio
        
        return target_stem, everything_else
    
    def apply_separation(self, stem_name: str, request: Optional[TwoStemSeparationRequest] = None) -> 'AudioProcessor':
        """
        Apply separation and update the processor's audio with the specified stem.
        
        Args:
            stem_name: Name of the stem to keep ("vocals", "drums", "bass", etc.)
            request: Parameters for separation
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = TwoStemSeparationRequest(stem=stem_name)
        else:
            request.stem = stem_name
            
        # Get the requested stem
        target_stem, _ = self.two_stems(request)
        
        # Set the processor's audio to the target stem
        self._processor.audio_data = target_stem
        
        # Return the processor for method chaining
        return self._processor
    
    def apply_vocal_removal(self, request: Optional[TwoStemSeparationRequest] = None) -> 'AudioProcessor':
        """
        Remove vocals from the audio (karaoke effect).
        
        Args:
            request: Parameters for vocal separation
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = TwoStemSeparationRequest(stem="vocals")
        else:
            request.stem = "vocals"
            
        # Get the vocals and accompaniment
        _, accompaniment = self.two_stems(request)
        
        # Set the processor's audio to the accompaniment (no vocals)
        self._processor.audio_data = accompaniment
        
        # Return the processor for method chaining
        return self._processor
    
    def available_models(self) -> List[str]:
        """
        Get a list of available Demucs models.
        
        Returns:
            List of available model names
        """
        # Common model names in Demucs
        return [
            "htdemucs",          # Default Hybrid Transformer Demucs
            "htdemucs_ft",       # Fine-tuned Hybrid Transformer Demucs
            "htdemucs_6s",       # 6-stem version with guitar and piano
            "hdemucs_mmi",       # Hybrid Demucs v3
            "mdx",               # MDX challenge winning model on track A
            "mdx_extra",         # MDX challenge model on track B
            "mdx_q",             # Quantized version of mdx
            "mdx_extra_q"        # Quantized version of mdx_extra
        ]
