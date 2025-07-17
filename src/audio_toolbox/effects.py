import numpy as np
from pedalboard import Pedalboard, Reverb, Delay, Chorus, Phaser, Compressor, Gain, Distortion
from pedalboard import Limiter, LowpassFilter, HighpassFilter, PitchShift, Clipping, NoiseGate
from typing import Optional, Tuple, Union, Dict, Any, List, TYPE_CHECKING
from pydantic import BaseModel, Field

# Type checking imports
if TYPE_CHECKING:
    from audio_toolbox.processor import AudioProcessor

# Request classes for different effects
class ReverbRequest(BaseModel):
    """Parameters for reverb effect"""
    room_size: float = Field(0.5, description="Size of the reverb room [0.0, 1.0]")
    damping: float = Field(0.5, description="Damping of high frequencies [0.0, 1.0]")
    wet_level: float = Field(0.33, description="Amount of wet signal [0.0, 1.0]")
    dry_level: float = Field(0.4, description="Amount of dry signal [0.0, 1.0]")
    width: float = Field(1.0, description="Stereo width [0.0, 1.0]")
    freeze_mode: float = Field(0.0, description="Freeze mode [0.0, 1.0]")

class DelayRequest(BaseModel):
    """Parameters for delay effect"""
    delay_seconds: float = Field(0.5, description="Delay time in seconds [0.0, 10.0]")
    feedback: float = Field(0.0, description="Feedback amount [0.0, 1.0]")
    mix: float = Field(0.5, description="Dry/wet mix [0.0, 1.0]")

class ChorusRequest(BaseModel):
    """Parameters for chorus effect"""
    rate_hz: float = Field(1.0, description="Rate of modulation in Hz [0.0, 100.0]")
    depth: float = Field(0.25, description="Depth of modulation [0.0, 1.0]")
    centre_delay_ms: float = Field(7.0, description="Centre delay in ms [1.0, 100.0]")
    feedback: float = Field(0.0, description="Feedback [0.0, 0.95]")
    mix: float = Field(0.5, description="Dry/wet mix [0.0, 1.0]")

class PhaserRequest(BaseModel):
    """Parameters for phaser effect"""
    rate_hz: float = Field(1.0, description="Rate of modulation in Hz [0.0, 10.0]")
    depth: float = Field(0.5, description="Depth of modulation [0.0, 1.0]")
    centre_frequency_hz: float = Field(1300.0, description="Centre frequency in Hz [20.0, 20000.0]")
    feedback: float = Field(0.0, description="Feedback [0.0, 0.95]")
    mix: float = Field(0.5, description="Dry/wet mix [0.0, 1.0]")

class CompressorRequest(BaseModel):
    """Parameters for compressor effect"""
    threshold_db: float = Field(-10.0, description="Threshold level in dB [-100.0, 0.0]")
    ratio: float = Field(2.0, description="Compression ratio [1.0, 20.0]")
    attack_ms: float = Field(1.0, description="Attack time in ms [0.1, 100.0]")
    release_ms: float = Field(100.0, description="Release time in ms [10.0, 1000.0]")

class DistortionRequest(BaseModel):
    """Parameters for distortion effect"""
    drive_db: float = Field(25.0, description="Drive amount in dB [0.0, 40.0]")

class GainRequest(BaseModel):
    """Parameters for gain effect"""
    gain_db: float = Field(0.0, description="Gain amount in dB [-60.0, 40.0]")

class PitchShiftRequest(BaseModel):
    """Parameters for pitch shift effect"""
    semitones: float = Field(0.0, description="Number of semitones to shift pitch [-12.0, 12.0]")

class FilterRequest(BaseModel):
    """Parameters for filter effects"""
    cutoff_frequency_hz: float = Field(1000.0, description="Cutoff frequency in Hz [20.0, 20000.0]")
    resonance: Optional[float] = Field(None, description="Resonance (Q factor) [0.1, 10.0]")

class NoiseGateRequest(BaseModel):
    """Parameters for noise gate effect"""
    threshold_db: float = Field(-100.0, description="Threshold level in dB [-100.0, 0.0]")
    ratio: float = Field(10.0, description="Ratio [1.0, 20.0]")
    attack_ms: float = Field(1.0, description="Attack time in ms [0.1, 1000.0]")
    release_ms: float = Field(100.0, description="Release time in ms [1.0, 1000.0]")

class LimiterRequest(BaseModel):
    """Parameters for limiter effect"""
    threshold_db: float = Field(-10.0, description="Threshold level in dB [-100.0, 0.0]")
    release_ms: float = Field(100.0, description="Release time in ms [1.0, 1000.0]")

class ChainRequest(BaseModel):
    """Parameters for effect chain"""
    effects: List[Dict[str, Any]] = Field([], description="List of effects in the chain with their parameters")

class AudioEffects:
    """
    Audio effects processing using pedalboard library.
    
    This class provides methods for applying various audio effects.
    """
    
    def __init__(self, processor: 'AudioProcessor'):
        """
        Initialize the effects processor with a reference to the AudioProcessor.
        
        Args:
            processor: The parent AudioProcessor instance
        """
        self._processor = processor
    
    def _check_audio(self):
        """Check that audio data is available for processing"""
        if self._processor.audio_data is None or self._processor.sample_rate is None:
            raise ValueError("No audio data available for processing")
    
    def _process_with_effect(self, effect):
        """Process audio with a single effect"""
        self._check_audio()
        
        # Create a Pedalboard with just this effect
        board = Pedalboard([effect])
        
        # Process each channel separately and return a new array
        processed_audio = np.zeros_like(self._processor.audio_data)
        for i in range(self._processor.audio_data.shape[0]):
            processed_audio[i] = board(self._processor.audio_data[i], self._processor.sample_rate)
            
        # Create a new processor with the processed audio
        self._processor.audio_data = processed_audio
        
        # Return the processor for chaining
        return self._processor
    
    def reverb(self, request: Optional[ReverbRequest] = None) -> 'AudioProcessor':
        """
        Apply reverb effect to the audio.
        
        Args:
            request: Parameters for the reverb effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = ReverbRequest()
            
        effect = Reverb(
            room_size=request.room_size,
            damping=request.damping,
            wet_level=request.wet_level,
            dry_level=request.dry_level,
            width=request.width,
            freeze_mode=request.freeze_mode
        )
        
        return self._process_with_effect(effect)
    
    def delay(self, request: Optional[DelayRequest] = None) -> 'AudioProcessor':
        """
        Apply delay effect to the audio.
        
        Args:
            request: Parameters for the delay effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = DelayRequest()
            
        effect = Delay(
            delay_seconds=request.delay_seconds,
            feedback=request.feedback,
            mix=request.mix
        )
        
        return self._process_with_effect(effect)
    
    def chorus(self, request: Optional[ChorusRequest] = None) -> 'AudioProcessor':
        """
        Apply chorus effect to the audio.
        
        Args:
            request: Parameters for the chorus effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = ChorusRequest()
            
        effect = Chorus(
            rate_hz=request.rate_hz,
            depth=request.depth,
            centre_delay_ms=request.centre_delay_ms,
            feedback=request.feedback,
            mix=request.mix
        )
        
        return self._process_with_effect(effect)
    
    def phaser(self, request: Optional[PhaserRequest] = None) -> 'AudioProcessor':
        """
        Apply phaser effect to the audio.
        
        Args:
            request: Parameters for the phaser effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = PhaserRequest()
            
        effect = Phaser(
            rate_hz=request.rate_hz,
            depth=request.depth,
            centre_frequency_hz=request.centre_frequency_hz,
            feedback=request.feedback,
            mix=request.mix
        )
        
        return self._process_with_effect(effect)
    
    def compressor(self, request: Optional[CompressorRequest] = None) -> 'AudioProcessor':
        """
        Apply compressor effect to the audio.
        
        Args:
            request: Parameters for the compressor effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = CompressorRequest()
            
        effect = Compressor(
            threshold_db=request.threshold_db,
            ratio=request.ratio,
            attack_ms=request.attack_ms,
            release_ms=request.release_ms
        )
        
        return self._process_with_effect(effect)
    
    def distortion(self, request: Optional[DistortionRequest] = None) -> 'AudioProcessor':
        """
        Apply distortion effect to the audio.
        
        Args:
            request: Parameters for the distortion effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = DistortionRequest()
            
        effect = Distortion(drive_db=request.drive_db)
        
        return self._process_with_effect(effect)
    
    def gain(self, request: Optional[GainRequest] = None) -> 'AudioProcessor':
        """
        Apply gain effect to the audio.
        
        Args:
            request: Parameters for the gain effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = GainRequest()
            
        effect = Gain(gain_db=request.gain_db)
        
        return self._process_with_effect(effect)
    
    def pitch_shift(self, request: Optional[PitchShiftRequest] = None) -> 'AudioProcessor':
        """
        Apply pitch shift effect to the audio.
        
        Args:
            request: Parameters for the pitch shift effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = PitchShiftRequest()
            
        effect = PitchShift(semitones=request.semitones)
        
        return self._process_with_effect(effect)
    
    def lowpass_filter(self, request: Optional[FilterRequest] = None) -> 'AudioProcessor':
        """
        Apply low-pass filter effect to the audio.
        
        Args:
            request: Parameters for the filter effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = FilterRequest()
            
        effect = LowpassFilter(cutoff_frequency_hz=request.cutoff_frequency_hz)
        
        return self._process_with_effect(effect)
    
    def highpass_filter(self, request: Optional[FilterRequest] = None) -> 'AudioProcessor':
        """
        Apply high-pass filter effect to the audio.
        
        Args:
            request: Parameters for the filter effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = FilterRequest()
            
        effect = HighpassFilter(cutoff_frequency_hz=request.cutoff_frequency_hz)
        
        return self._process_with_effect(effect)
    
    def noise_gate(self, request: Optional[NoiseGateRequest] = None) -> 'AudioProcessor':
        """
        Apply noise gate effect to the audio.
        
        Args:
            request: Parameters for the noise gate effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = NoiseGateRequest()
            
        effect = NoiseGate(
            threshold_db=request.threshold_db,
            ratio=request.ratio,
            attack_ms=request.attack_ms,
            release_ms=request.release_ms
        )
        
        return self._process_with_effect(effect)
    
    def limiter(self, request: Optional[LimiterRequest] = None) -> 'AudioProcessor':
        """
        Apply limiter effect to the audio.
        
        Args:
            request: Parameters for the limiter effect
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        if request is None:
            request = LimiterRequest()
            
        effect = Limiter(
            threshold_db=request.threshold_db,
            release_ms=request.release_ms
        )
        
        return self._process_with_effect(effect)
    
    def chain(self, request: ChainRequest) -> 'AudioProcessor':
        """
        Apply a chain of effects to the audio.
        
        Args:
            request: Parameters for the effect chain
            
        Returns:
            The AudioProcessor instance with processed audio
        """
        self._check_audio()
        
        # Create an empty pedalboard
        board = Pedalboard()
        
        # Add each effect to the pedalboard
        for effect_dict in request.effects:
            effect_type = effect_dict.pop('type')
            
            if effect_type == 'reverb':
                board.append(Reverb(**effect_dict))
            elif effect_type == 'delay':
                board.append(Delay(**effect_dict))
            elif effect_type == 'chorus':
                board.append(Chorus(**effect_dict))
            elif effect_type == 'phaser':
                board.append(Phaser(**effect_dict))
            elif effect_type == 'compressor':
                board.append(Compressor(**effect_dict))
            elif effect_type == 'distortion':
                board.append(Distortion(**effect_dict))
            elif effect_type == 'gain':
                board.append(Gain(**effect_dict))
            elif effect_type == 'pitch_shift':
                board.append(PitchShift(**effect_dict))
            elif effect_type == 'lowpass_filter':
                board.append(LowpassFilter(**effect_dict))
            elif effect_type == 'highpass_filter':
                board.append(HighpassFilter(**effect_dict))
            elif effect_type == 'noise_gate':
                board.append(NoiseGate(**effect_dict))
            elif effect_type == 'limiter':
                board.append(Limiter(**effect_dict))
            elif effect_type == 'clipping':
                board.append(Clipping(**effect_dict))
        
        # Process each channel separately and return a new array
        processed_audio = np.zeros_like(self._processor.audio_data)
        for i in range(self._processor.audio_data.shape[0]):
            processed_audio[i] = board(self._processor.audio_data[i], self._processor.sample_rate)
            
        # Create a new processor with the processed audio
        self._processor.audio_data = processed_audio
        
        # Return the processor for chaining
        return self._processor
