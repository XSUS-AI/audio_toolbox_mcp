import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, Union, Dict, Any, List
from pathlib import Path

from audio_toolbox.analyze import AudioAnalyzer
from audio_toolbox.effects import AudioEffects
from audio_toolbox.separate import AudioSeparator
from audio_toolbox.transform import AudioTransformer

class AudioProcessor:
    """
    Main class for audio processing, providing access to all audio toolbox functionality.
    
    This class loads audio files and provides interfaces to analysis, effects, separation, 
    and transformation modules.
    """
    
    def __init__(self, file_path: Optional[Union[str, Path]] = None, audio_data: Optional[np.ndarray] = None, 
                 sample_rate: Optional[int] = None):
        """
        Initialize an AudioProcessor with either a file path or audio data.
        
        Args:
            file_path: Path to an audio file to load
            audio_data: NumPy array containing audio data
            sample_rate: Sample rate of the audio data (required if audio_data is provided)
        """
        self._audio_data = None
        self._sample_rate = None
        
        # Load from file if provided
        if file_path is not None:
            self.load_file(file_path)
        # Or use provided audio data
        elif audio_data is not None:
            if sample_rate is None:
                raise ValueError("Sample rate must be provided when audio data is provided directly")
            self._audio_data = audio_data
            self._sample_rate = sample_rate
        
        # Create module interfaces
        self._analyzer = None
        self._effects = None
        self._separator = None
        self._transformer = None
    
    @property
    def analyze(self) -> AudioAnalyzer:
        """Interface to audio analysis features"""
        if self._analyzer is None:
            self._analyzer = AudioAnalyzer(self)
        return self._analyzer
    
    @property
    def effects(self) -> AudioEffects:
        """Interface to audio effects"""
        if self._effects is None:
            self._effects = AudioEffects(self)
        return self._effects
    
    @property
    def separate(self) -> AudioSeparator:
        """Interface to audio source separation"""
        if self._separator is None:
            self._separator = AudioSeparator(self)
        return self._separator
    
    @property
    def transform(self) -> AudioTransformer:
        """Interface to audio transformations"""
        if self._transformer is None:
            self._transformer = AudioTransformer(self)
        return self._transformer
    
    @property
    def audio_data(self) -> np.ndarray:
        """Get the current audio data"""
        return self._audio_data
    
    @audio_data.setter
    def audio_data(self, data: np.ndarray) -> None:
        """Set the current audio data"""
        self._audio_data = data
    
    @property
    def sample_rate(self) -> int:
        """Get the current sample rate"""
        return self._sample_rate
    
    @sample_rate.setter
    def sample_rate(self, rate: int) -> None:
        """Set the current sample rate"""
        self._sample_rate = rate
    
    @property
    def duration(self) -> float:
        """Get the duration of the audio in seconds"""
        if self._audio_data is None or self._sample_rate is None:
            return 0.0
        return len(self._audio_data) / self._sample_rate
    
    @property
    def num_channels(self) -> int:
        """Get the number of audio channels"""
        if self._audio_data is None:
            return 0
        if self._audio_data.ndim == 1:
            return 1
        return self._audio_data.shape[0] if self._audio_data.shape[0] < self._audio_data.shape[1] else self._audio_data.shape[1]
    
    def load_file(self, file_path: Union[str, Path]) -> None:
        """Load audio from a file"""
        self._audio_data, self._sample_rate = librosa.load(file_path, sr=None, mono=False)
        # Ensure 2D array for consistency (channels, samples)
        if self._audio_data.ndim == 1:
            self._audio_data = self._audio_data[np.newaxis, :]
    
    def save(self, file_path: Union[str, Path], format: Optional[str] = None) -> None:
        """Save the current audio data to a file"""
        if self._audio_data is None or self._sample_rate is None:
            raise ValueError("No audio data to save")
            
        # Convert from (channels, samples) to (samples, channels) for soundfile
        audio_to_save = self._audio_data.T if self._audio_data.ndim > 1 else self._audio_data
        
        sf.write(file_path, audio_to_save, self._sample_rate, format=format)
    
    def reset(self) -> None:
        """Clear all audio data"""
        self._audio_data = None
        self._sample_rate = None
        self._analyzer = None
        self._effects = None
        self._separator = None
        self._transformer = None
