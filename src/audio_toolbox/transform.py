import numpy as np
import librosa
import audioflux as af
from pedalboard import time_stretch, PitchShift, Resample
from typing import Optional, Tuple, Union, Dict, Any, List, TYPE_CHECKING
from pydantic import BaseModel, Field

# Type checking imports
if TYPE_CHECKING:
    from audio_toolbox.processor import AudioProcessor

class TimeStretchRequest(BaseModel):
    """Parameters for time stretching"""
    stretch_factor: float = Field(1.0, description="Stretch factor (>1 speeds up, <1 slows down)")
    high_quality: bool = Field(True, description="Whether to use high-quality stretching")
    preserve_formants: bool = Field(False, description="Whether to preserve formants during stretching")
    transient_mode: str = Field("crisp", description="How to handle transients: 'crisp', 'mixed', or 'smooth'")
    retain_phase_continuity: bool = Field(True, description="Whether to maintain phase continuity")

class PitchShiftRequest(BaseModel):
    """Parameters for pitch shifting"""
    semitones: float = Field(0.0, description="Number of semitones to shift pitch")
    high_quality: bool = Field(True, description="Whether to use high-quality processing")
    preserve_formants: bool = Field(False, description="Whether to preserve formants during shifting")

class TimeAndPitchRequest(BaseModel):
    """Parameters for combined time stretching and pitch shifting"""
    stretch_factor: float = Field(1.0, description="Stretch factor (>1 speeds up, <1 slows down)")
    pitch_shift_semitones: float = Field(0.0, description="Number of semitones to shift pitch")
    high_quality: bool = Field(True, description="Whether to use high-quality processing")
    preserve_formants: bool = Field(False, description="Whether to preserve formants")
    transient_mode: str = Field("crisp", description="How to handle transients: 'crisp', 'mixed', or 'smooth'")

class ResampleRequest(BaseModel):
    """Parameters for resampling"""
    target_sample_rate: int = Field(44100, description="Target sample rate in Hz")
    quality: str = Field("medium", description="Resampling quality: high, medium, or low")

class AudioTransformer:
    """
    Audio transformation functionality combining time and frequency domain transformations.
    
    This class provides methods for transforming audio signals in various ways.
    """
    
    def __init__(self, processor: 'AudioProcessor'):
        """
        Initialize the transformer with a reference to the AudioProcessor.
        
        Args:
            processor: The parent AudioProcessor instance
        """
        self._processor = processor
    
    def _check_audio(self):
        """Check that audio data is available for transformation"""
        if self._processor.audio_data is None or self._processor.sample_rate is None:
            raise ValueError("No audio data available for transformation")
    
    def time_stretch(self, request: Optional[TimeStretchRequest] = None) -> 'AudioProcessor':
        """
        Stretch or compress audio in time without affecting pitch.
        
        Args:
            request: Parameters for time stretching
            
        Returns:
            The AudioProcessor instance with transformed audio
        """
        self._check_audio()
        
        if request is None:
            request = TimeStretchRequest()
        
        # Ensure the stretch factor is valid
        if request.stretch_factor <= 0:
            raise ValueError("Stretch factor must be greater than 0")
        
        # Process each channel separately
        processed_audio = []
        for channel in range(self._processor.audio_data.shape[0]):
            # Use Pedalboard's time_stretch function
            processed = time_stretch(
                self._processor.audio_data[channel], 
                self._processor.sample_rate,
                request.stretch_factor,
                high_quality=request.high_quality,
                preserve_formants=request.preserve_formants,
                transient_mode=request.transient_mode,
                retain_phase_continuity=request.retain_phase_continuity
            )
            processed_audio.append(processed)
        
        # Stack the processed channels and update the processor's audio
        self._processor.audio_data = np.vstack(processed_audio)
        
        # Return the processor for method chaining
        return self._processor
    
    def pitch_shift(self, request: Optional[PitchShiftRequest] = None) -> 'AudioProcessor':
        """
        Shift the pitch of audio without affecting its duration.
        
        Args:
            request: Parameters for pitch shifting
            
        Returns:
            The AudioProcessor instance with transformed audio
        """
        self._check_audio()
        
        if request is None:
            request = PitchShiftRequest()
        
        # Use PitchShift from pedalboard for simple shifts
        if not request.preserve_formants:
            # Create the pitch shift effect
            effect = PitchShift(semitones=request.semitones)
            
            # Process each channel separately
            processed_audio = np.zeros_like(self._processor.audio_data)
            for i in range(self._processor.audio_data.shape[0]):
                processed_audio[i] = effect(self._processor.audio_data[i], self._processor.sample_rate)
                
            self._processor.audio_data = processed_audio
        else:
            # For formant-preserved pitch shifting, use time_stretch with pitch shift
            processed_audio = []
            for channel in range(self._processor.audio_data.shape[0]):
                # Use Pedalboard's time_stretch function with pitch shift
                processed = time_stretch(
                    self._processor.audio_data[channel], 
                    self._processor.sample_rate,
                    1.0,  # No time stretching, just pitch shift
                    pitch_shift_in_semitones=request.semitones,
                    high_quality=request.high_quality,
                    preserve_formants=True
                )
                processed_audio.append(processed)
            
            # Stack the processed channels and update the processor's audio
            self._processor.audio_data = np.vstack(processed_audio)
        
        # Return the processor for method chaining
        return self._processor
    
    def time_and_pitch(self, request: Optional[TimeAndPitchRequest] = None) -> 'AudioProcessor':
        """
        Transform audio by changing both time and pitch simultaneously.
        
        Args:
            request: Parameters for time and pitch transformation
            
        Returns:
            The AudioProcessor instance with transformed audio
        """
        self._check_audio()
        
        if request is None:
            request = TimeAndPitchRequest()
        
        # Ensure the stretch factor is valid
        if request.stretch_factor <= 0:
            raise ValueError("Stretch factor must be greater than 0")
        
        # Process each channel separately
        processed_audio = []
        for channel in range(self._processor.audio_data.shape[0]):
            # Use Pedalboard's time_stretch function with pitch shift
            processed = time_stretch(
                self._processor.audio_data[channel], 
                self._processor.sample_rate,
                request.stretch_factor,
                pitch_shift_in_semitones=request.pitch_shift_semitones,
                high_quality=request.high_quality,
                preserve_formants=request.preserve_formants,
                transient_mode=request.transient_mode
            )
            processed_audio.append(processed)
        
        # Stack the processed channels and update the processor's audio
        self._processor.audio_data = np.vstack(processed_audio)
        
        # Return the processor for method chaining
        return self._processor
    
    def resample(self, request: Optional[ResampleRequest] = None) -> 'AudioProcessor':
        """
        Resample audio to a different sample rate.
        
        Args:
            request: Parameters for resampling
            
        Returns:
            The AudioProcessor instance with resampled audio
        """
        self._check_audio()
        
        if request is None:
            request = ResampleRequest()
        
        # No resampling needed if already at target rate
        if self._processor.sample_rate == request.target_sample_rate:
            return self._processor
        
        # Map quality string to pedalboard.Resample.Quality enum
        quality_map = {
            "high": "WindowedSinc64",
            "medium": "WindowedSinc32",
            "low": "WindowedSinc16"
        }
        quality = quality_map.get(request.quality.lower(), "WindowedSinc32")
        
        # Create a resample effect
        effect = Resample(target_sample_rate=request.target_sample_rate, quality=quality)
        
        # Process each channel separately
        processed_audio = np.zeros((self._processor.audio_data.shape[0], 
                                   int(self._processor.audio_data.shape[1] * 
                                       request.target_sample_rate / self._processor.sample_rate)))
        
        for i in range(self._processor.audio_data.shape[0]):
            processed_audio[i] = effect(self._processor.audio_data[i], self._processor.sample_rate)
        
        # Update the processor's audio and sample rate
        self._processor.audio_data = processed_audio
        self._processor.sample_rate = request.target_sample_rate
        
        # Return the processor for method chaining
        return self._processor
    
    def normalize(self, target_db: float = -1.0) -> 'AudioProcessor':
        """
        Normalize the audio to have a specific peak level.
        
        Args:
            target_db: Target peak level in dB (0 = maximum digital value)
            
        Returns:
            The AudioProcessor instance with normalized audio
        """
        self._check_audio()
        
        # Find the maximum absolute amplitude
        max_amp = np.max(np.abs(self._processor.audio_data))
        
        if max_amp == 0:
            # Audio is completely silent, nothing to normalize
            return self._processor
        
        # Calculate target amplitude (0 dB = 1.0 amplitude)
        target_amp = 10 ** (target_db / 20.0)
        
        # Calculate gain factor
        gain = target_amp / max_amp
        
        # Apply gain
        self._processor.audio_data *= gain
        
        # Return the processor for method chaining
        return self._processor
    
    def reverse(self) -> 'AudioProcessor':
        """
        Reverse the audio in time.
        
        Returns:
            The AudioProcessor instance with reversed audio
        """
        self._check_audio()
        
        # Reverse each channel
        self._processor.audio_data = np.flip(self._processor.audio_data, axis=1)
        
        # Return the processor for method chaining
        return self._processor
    
    def fade_in(self, duration_seconds: float = 1.0) -> 'AudioProcessor':
        """
        Apply a fade-in effect to the audio.
        
        Args:
            duration_seconds: Duration of the fade-in in seconds
            
        Returns:
            The AudioProcessor instance with fade-in applied
        """
        self._check_audio()
        
        # Calculate the number of samples for the fade
        num_samples = int(duration_seconds * self._processor.sample_rate)
        
        # Ensure we don't try to fade more than the audio length
        if num_samples > self._processor.audio_data.shape[1]:
            num_samples = self._processor.audio_data.shape[1]
        
        if num_samples > 0:
            # Create a linear fade-in envelope
            fade_in = np.linspace(0, 1, num_samples)
            
            # Apply the fade-in to each channel
            for i in range(self._processor.audio_data.shape[0]):
                self._processor.audio_data[i, :num_samples] *= fade_in
        
        # Return the processor for method chaining
        return self._processor
    
    def fade_out(self, duration_seconds: float = 1.0) -> 'AudioProcessor':
        """
        Apply a fade-out effect to the audio.
        
        Args:
            duration_seconds: Duration of the fade-out in seconds
            
        Returns:
            The AudioProcessor instance with fade-out applied
        """
        self._check_audio()
        
        # Calculate the number of samples for the fade
        num_samples = int(duration_seconds * self._processor.sample_rate)
        
        # Ensure we don't try to fade more than the audio length
        if num_samples > self._processor.audio_data.shape[1]:
            num_samples = self._processor.audio_data.shape[1]
        
        if num_samples > 0:
            # Create a linear fade-out envelope
            fade_out = np.linspace(1, 0, num_samples)
            
            # Apply the fade-out to each channel
            for i in range(self._processor.audio_data.shape[0]):
                self._processor.audio_data[i, -num_samples:] *= fade_out
        
        # Return the processor for method chaining
        return self._processor
    
    def trim_silence(self, threshold_db: float = -60.0, pad_seconds: float = 0.1) -> 'AudioProcessor':
        """
        Trim silence from the beginning and end of the audio.
        
        Args:
            threshold_db: Threshold for silence detection in dB
            pad_seconds: Seconds of padding to leave around non-silent regions
            
        Returns:
            The AudioProcessor instance with trimmed audio
        """
        self._check_audio()
        
        # Convert mono audio for detection
        mono_audio = np.mean(self._processor.audio_data, axis=0) if self._processor.audio_data.shape[0] > 1 else self._processor.audio_data[0]
        
        # Trim using librosa
        trimmed_audio, trim_indices = librosa.effects.trim(
            mono_audio, 
            top_db=-threshold_db,  # Convert from negative dB to positive dB for librosa
            frame_length=2048,
            hop_length=512
        )
        
        # Add padding (if requested)
        pad_samples = int(pad_seconds * self._processor.sample_rate)
        start_idx = max(0, trim_indices[0] - pad_samples)
        end_idx = min(self._processor.audio_data.shape[1], trim_indices[1] + pad_samples)
        
        # Trim all channels
        self._processor.audio_data = self._processor.audio_data[:, start_idx:end_idx]
        
        # Return the processor for method chaining
        return self._processor
