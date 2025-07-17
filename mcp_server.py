"""MCP Server for audio analysis and transformations."""

import os
import json
import base64
import tempfile
import logging
import logging.handlers
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Union, List, Dict, Any, Literal, Tuple
from pathlib import Path
import soundfile as sf
import librosa

from pydantic import BaseModel, Field, field_validator
from mcp.server.fastmcp import FastMCP, Context, Image

# Set up logging
logger = logging.getLogger("audio_toolbox")
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Set up rotating file handler with compression
handler = logging.handlers.RotatingFileHandler(
    log_dir / "audio_toolbox.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8"
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(console_handler)

# Import audio_toolbox after setting up logging
from src.audio_toolbox import AudioProcessor
from src.audio_toolbox.analyze import (
    MelSpectrogramRequest, ChromagramRequest, OnsetDetectionRequest,
    BeatTrackingRequest, SpectralFeaturesRequest, PitchTrackingRequest
)
from src.audio_toolbox.effects import (
    ReverbRequest, DelayRequest, ChorusRequest, PhaserRequest,
    CompressorRequest, DistortionRequest, GainRequest, PitchShiftRequest as EffectPitchShiftRequest,
    FilterRequest, NoiseGateRequest, LimiterRequest, ChainRequest
)
from src.audio_toolbox.transform import (
    TimeStretchRequest, PitchShiftRequest as TransformPitchShiftRequest, 
    TimeAndPitchRequest, ResampleRequest
)
from src.audio_toolbox.separate import (
    SeparationRequest, TwoStemSeparationRequest
)

# Audio Registry for in-memory audio data management
class AudioRegistry:
    """Registry for storing audio data with access via IDs."""
    
    def __init__(self):
        self.audio_registry = {}  # Maps IDs to (audio_data, sample_rate, metadata)
        self.counter = 0  # For generating unique IDs
    
    def register(self, audio_data, sample_rate, name=None, metadata=None):
        """Register audio data and return a unique ID."""
        if name and name in self.audio_registry:
            # If named entry exists, replace it
            audio_id = name
        else:
            # Generate a unique ID if no name provided
            self.counter += 1
            audio_id = name or f"audio_{self.counter}"
        
        self.audio_registry[audio_id] = {
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "metadata": metadata or {},
            "created": datetime.now().isoformat()
        }
        return audio_id
    
    def get(self, audio_id):
        """Retrieve audio data by ID."""
        if audio_id not in self.audio_registry:
            raise ValueError(f"No audio data found with ID '{audio_id}'")
        return self.audio_registry[audio_id]
    
    def list_entries(self):
        """List all registered audio entries with metadata."""
        return {
            audio_id: {
                "sample_rate": entry["sample_rate"],
                "created": entry["created"],
                "metadata": entry["metadata"]
            }
            for audio_id, entry in self.audio_registry.items()
        }
    
    def remove(self, audio_id):
        """Remove audio data from registry."""
        if audio_id in self.audio_registry:
            del self.audio_registry[audio_id]
            return True
        return False
    
    def clear(self):
        """Clear all audio data from registry."""
        self.audio_registry.clear()
        self.counter = 0

# Base Models for Requests and Responses
class AudioFileRequest(BaseModel):
    """Request to load an audio file."""
    file_path: str = Field(..., description="Path to the audio file to load")

class AudioDataRequest(BaseModel):
    """Request to process audio data directly."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(..., description="Sample rate of the audio data in Hz")
    format: Optional[str] = Field(None, description="Format of the audio data (wav, mp3, etc.)")
    
    @field_validator('audio_data')
    def validate_audio_data(cls, v):
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("audio_data must be a valid base64 encoded string")

class SaveFileRequest(BaseModel):
    """Request to save audio to a file."""
    file_path: str = Field(..., description="Path where to save the audio file")
    format: Optional[str] = Field(None, description="Format to save as (wav, mp3, etc.)")

class AudioInfoResponse(BaseModel):
    """Response with audio information."""
    duration: float = Field(..., description="Duration of the audio in seconds")
    sample_rate: int = Field(..., description="Sample rate of the audio in Hz")
    num_channels: int = Field(..., description="Number of audio channels")

class SpectrogramResponse(BaseModel):
    """Response with spectrogram data."""
    data: str = Field(..., description="Base64 encoded spectrogram image")
    format: str = Field("png", description="Image format")

class OnsetResponse(BaseModel):
    """Response with onset detection results."""
    onsets: List[float] = Field(..., description="List of onset times in seconds")

class BeatResponse(BaseModel):
    """Response with beat tracking results."""
    beats: List[float] = Field(..., description="List of beat times in seconds")
    tempo: float = Field(..., description="Detected tempo in BPM")

class SpectralFeaturesResponse(BaseModel):
    """Response with spectral features."""
    features: Dict[str, List[float]] = Field(..., description="Dictionary of spectral features")

class PitchResponse(BaseModel):
    """Response with pitch tracking results."""
    times: List[float] = Field(..., description="Time points in seconds")
    frequencies: List[float] = Field(..., description="Detected frequencies in Hz")

class AudioProcessingResponse(BaseModel):
    """Response after audio processing."""
    success: bool = Field(True, description="Whether the operation was successful")
    message: str = Field("Operation completed successfully", description="Status message")

class AudioOutputResponse(BaseModel):
    """Response with processed audio data."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(..., description="Sample rate of the audio in Hz")
    format: str = Field("wav", description="Format of the audio data")

# Audio Registry specific models
class AudioRegistryEntryInfo(BaseModel):
    """Information about an entry in the audio registry."""
    audio_id: str = Field(..., description="ID of the registered audio data")
    sample_rate: int = Field(..., description="Sample rate of the audio")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    created: str = Field(..., description="ISO timestamp when the audio was registered")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata about the audio")

class AudioRegistryListResponse(BaseModel):
    """Response listing all entries in the audio registry."""
    entries: List[AudioRegistryEntryInfo] = Field(..., description="List of registry entries")

class AudioRegistryRequest(BaseModel):
    """Request to retrieve audio from the registry."""
    audio_id: str = Field(..., description="ID of the registered audio data")

class RegisteredAudioResponse(BaseModel):
    """Response with a reference to registered audio data."""
    audio_id: str = Field(..., description="ID of the registered audio data")
    sample_rate: int = Field(..., description="Sample rate of the audio in Hz")
    duration: Optional[float] = Field(None, description="Duration of the audio in seconds")

# Updated separation response model with registry references
class SeparationRegistryResponse(BaseModel):
    """Response with audio separation results as registry references."""
    stem_ids: Dict[str, str] = Field(..., description="Dictionary of stem names to registry IDs")
    sample_rate: int = Field(..., description="Sample rate of the stems in Hz")

class VocalsRegistryResponse(BaseModel):
    """Response with vocals separation results as registry references."""
    vocals_id: str = Field(..., description="Registry ID for vocals audio")
    vocals_path: str = Field(..., description="Path to saved vocals file")
    accompaniment_id: str = Field(..., description="Registry ID for accompaniment audio")
    accompaniment_path: str = Field(..., description="Path to saved accompaniment file")
    sample_rate: int = Field(..., description="Sample rate of the stems in Hz")

class TwoStemRegistryResponse(BaseModel):
    """Response with two-stem separation results as registry references."""
    target_stem_id: str = Field(..., description="Registry ID for target stem audio")
    target_stem_path: str = Field(..., description="Path to saved target stem file")
    remainder_id: str = Field(..., description="Registry ID for remainder audio")
    remainder_path: str = Field(..., description="Path to saved remainder file")
    sample_rate: int = Field(..., description="Sample rate of the stems in Hz")

# Create an MCP server
mcp = FastMCP("Audio Toolbox")

# Define the application state and lifespan
class AppContext:
    """Context for the audio processor."""
    def __init__(self):
        self.processor = None
        self.audio_registry = AudioRegistry()
        self.output_dir = Path("output")
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> None:
    """Manage application lifecycle."""
    # Initialize on startup
    context = AppContext()
    try:
        yield context
    finally:
        # Cleanup on shutdown
        if context.processor is not None:
            context.processor.reset()

# Pass lifespan to server
mcp = FastMCP("Audio Toolbox", lifespan=app_lifespan)

# Helper functions for encoding/decoding audio data
def encode_audio_data(audio_data: np.ndarray, sample_rate: int, format: str = "wav") -> str:
    """Convert audio data to base64 encoded string."""
    with tempfile.NamedTemporaryFile(suffix=f".{format}") as tmp:
        # Convert from (channels, samples) to (samples, channels) for soundfile
        audio_to_save = audio_data.T if audio_data.ndim > 1 else audio_data
        sf.write(tmp.name, audio_to_save, sample_rate, format=format)
        tmp.flush()
        tmp.seek(0)
        encoded = base64.b64encode(tmp.read()).decode('utf-8')
    return encoded

def decode_audio_data(audio_data_str: str) -> Tuple[np.ndarray, int]:
    """Convert base64 encoded string to audio data and sample rate."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(base64.b64decode(audio_data_str))
        tmp.flush()
        tmp_path = tmp.name
    
    try:
        audio_data, sample_rate = librosa.load(tmp_path, sr=None, mono=False)
        # Ensure 2D array for consistency (channels, samples)
        if audio_data.ndim == 1:
            audio_data = audio_data[np.newaxis, :]
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return audio_data, sample_rate

# Audio Registry Management Tools
@mcp.tool()
def list_audio_registry(ctx: Context) -> AudioRegistryListResponse:
    """List all entries in the audio registry."""
    registry = ctx.request_context.lifespan_context.audio_registry
    entries_data = registry.list_entries()
    
    entries = []
    for audio_id, info in entries_data.items():
        # Get duration if available
        duration = None
        processor = ctx.request_context.lifespan_context.processor
        if processor and audio_id == "current" and processor.audio_data is not None:
            duration = processor.duration
            
        entries.append(AudioRegistryEntryInfo(
            audio_id=audio_id,
            sample_rate=info["sample_rate"],
            duration=duration,
            created=info["created"],
            metadata=info["metadata"]
        ))
    
    return AudioRegistryListResponse(entries=entries)

@mcp.tool()
def load_audio_from_registry(request: AudioRegistryRequest, ctx: Context) -> AudioInfoResponse:
    """Load audio from the registry into the processor."""
    try:
        registry = ctx.request_context.lifespan_context.audio_registry
        audio_entry = registry.get(request.audio_id)
        
        processor = AudioProcessor(
            audio_data=audio_entry["audio_data"],
            sample_rate=audio_entry["sample_rate"]
        )
        ctx.request_context.lifespan_context.processor = processor
        
        # Register as current
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata=audio_entry["metadata"]
        )
        
        return AudioInfoResponse(
            duration=processor.duration,
            sample_rate=processor.sample_rate,
            num_channels=processor.num_channels
        )
    except Exception as e:
        logger.error(f"Error loading audio from registry: {str(e)}")
        raise ValueError(f"Failed to load audio from registry: {str(e)}")

@mcp.tool()
def remove_from_audio_registry(request: AudioRegistryRequest, ctx: Context) -> AudioProcessingResponse:
    """Remove an entry from the audio registry."""
    registry = ctx.request_context.lifespan_context.audio_registry
    if registry.remove(request.audio_id):
        return AudioProcessingResponse(
            success=True,
            message=f"Removed audio with ID '{request.audio_id}' from registry"
        )
    return AudioProcessingResponse(
        success=False,
        message=f"No audio found with ID '{request.audio_id}'"
    )

@mcp.tool()
def clear_audio_registry(ctx: Context) -> AudioProcessingResponse:
    """Clear all entries from the audio registry."""
    registry = ctx.request_context.lifespan_context.audio_registry
    registry.clear()
    return AudioProcessingResponse(
        success=True,
        message="Audio registry cleared"
    )

# Audio loading and info tools
@mcp.tool()
def load_audio_file(request: AudioFileRequest, ctx: Context) -> AudioInfoResponse:
    """Load an audio file and return information about it."""
    try:
        logger.info(f"Loading audio file: {request.file_path}")
        processor = AudioProcessor(request.file_path)
        ctx.request_context.lifespan_context.processor = processor
        
        # Register the loaded audio
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",  # Always use "current" for the active processor
            metadata={"source_file": request.file_path}
        )
        
        return AudioInfoResponse(
            duration=processor.duration,
            sample_rate=processor.sample_rate,
            num_channels=processor.num_channels
        )
    except Exception as e:
        logger.error(f"Error loading audio file: {str(e)}")
        raise ValueError(f"Failed to load audio file: {str(e)}")

@mcp.tool()
def load_audio_data(request: AudioDataRequest, ctx: Context) -> AudioInfoResponse:
    """Load audio data from base64 encoded string."""
    try:
        logger.info("Loading audio from provided data")
        audio_data, sample_rate = decode_audio_data(request.audio_data)
        
        processor = AudioProcessor(audio_data=audio_data, sample_rate=sample_rate)
        ctx.request_context.lifespan_context.processor = processor
        
        # Register the loaded audio
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"source": "loaded_data"}
        )
        
        return AudioInfoResponse(
            duration=processor.duration,
            sample_rate=processor.sample_rate,
            num_channels=processor.num_channels
        )
    except Exception as e:
        logger.error(f"Error loading audio data: {str(e)}")
        raise ValueError(f"Failed to load audio data: {str(e)}")

@mcp.tool()
def save_audio(request: SaveFileRequest, ctx: Context) -> AudioProcessingResponse:
    """Save the current audio to a file."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info(f"Saving audio to: {request.file_path}")
        processor.save(request.file_path, format=request.format)
        
        return AudioProcessingResponse(
            success=True,
            message=f"Audio saved to {request.file_path}"
        )
    except Exception as e:
        logger.error(f"Error saving audio: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to save audio: {str(e)}"
        )

@mcp.tool()
def get_audio_data(ctx: Context) -> AudioOutputResponse:
    """Get the current audio data as a base64 encoded string."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Retrieving audio data")
        encoded_data = encode_audio_data(processor.audio_data, processor.sample_rate)
        
        return AudioOutputResponse(
            audio_data=encoded_data,
            sample_rate=processor.sample_rate,
            format="wav"
        )
    except Exception as e:
        logger.error(f"Error getting audio data: {str(e)}")
        raise ValueError(f"Failed to get audio data: {str(e)}")

# Audio analysis tools
@mcp.tool()
def analyze_mel_spectrogram(request: Optional[MelSpectrogramRequest] = None, ctx: Context = None) -> SpectrogramResponse:
    """Generate a mel spectrogram from the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Generating mel spectrogram")
        # Generate the mel spectrogram
        mel_spec = processor.analyze.mel_spectrogram(request)
        
        # Convert to dB scale
        import librosa
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create an image of the spectrogram
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=processor.sample_rate, ax=ax)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        # Save figure to a bytes object
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            plt.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            tmp.seek(0)
            img_data = base64.b64encode(tmp.read()).decode('utf-8')
        
        return SpectrogramResponse(
            data=img_data,
            format="png"
        )
    except Exception as e:
        logger.error(f"Error generating mel spectrogram: {str(e)}")
        raise ValueError(f"Failed to generate mel spectrogram: {str(e)}")

@mcp.tool()
def analyze_chromagram(request: Optional[ChromagramRequest] = None, ctx: Context = None) -> SpectrogramResponse:
    """Generate a chromagram (pitch content) visualization from the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Generating chromagram")
        # Generate the chromagram
        chroma = processor.analyze.chromagram(request)
        
        # Create an image of the chromagram
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=processor.sample_rate, ax=ax)
        plt.colorbar()
        plt.title('Chromagram')
        
        # Save figure to a bytes object
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            plt.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            tmp.seek(0)
            img_data = base64.b64encode(tmp.read()).decode('utf-8')
        
        return SpectrogramResponse(
            data=img_data,
            format="png"
        )
    except Exception as e:
        logger.error(f"Error generating chromagram: {str(e)}")
        raise ValueError(f"Failed to generate chromagram: {str(e)}")

@mcp.tool()
def detect_onsets(request: Optional[OnsetDetectionRequest] = None, ctx: Context = None) -> OnsetResponse:
    """Detect onsets (note beginnings) in the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Detecting onsets")
        onsets = processor.analyze.detect_onsets(request)
        
        return OnsetResponse(
            onsets=onsets.tolist()
        )
    except Exception as e:
        logger.error(f"Error detecting onsets: {str(e)}")
        raise ValueError(f"Failed to detect onsets: {str(e)}")

@mcp.tool()
def track_beats(request: Optional[BeatTrackingRequest] = None, ctx: Context = None) -> BeatResponse:
    """Track beats and estimate tempo in the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Tracking beats")
        beats, tempo = processor.analyze.track_beats(request)
        
        return BeatResponse(
            beats=beats.tolist(),
            tempo=float(tempo)
        )
    except Exception as e:
        logger.error(f"Error tracking beats: {str(e)}")
        raise ValueError(f"Failed to track beats: {str(e)}")

@mcp.tool()
def extract_spectral_features(request: Optional[SpectralFeaturesRequest] = None, ctx: Context = None) -> SpectralFeaturesResponse:
    """Extract spectral features from the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Extracting spectral features")
        features = processor.analyze.spectral_features(request)
        
        # Convert numpy arrays to lists for JSON serialization
        features_dict = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                # Take mean across time dimension to get a summary feature
                features_dict[key] = value.mean(axis=1).tolist()
            else:
                features_dict[key] = value
        
        return SpectralFeaturesResponse(
            features=features_dict
        )
    except Exception as e:
        logger.error(f"Error extracting spectral features: {str(e)}")
        raise ValueError(f"Failed to extract spectral features: {str(e)}")

@mcp.tool()
def track_pitch(request: Optional[PitchTrackingRequest] = None, ctx: Context = None) -> PitchResponse:
    """Track pitch (fundamental frequency) in the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Tracking pitch")
        times, frequencies = processor.analyze.track_pitch(request)
        
        return PitchResponse(
            times=times.tolist(),
            frequencies=frequencies.tolist()
        )
    except Exception as e:
        logger.error(f"Error tracking pitch: {str(e)}")
        raise ValueError(f"Failed to track pitch: {str(e)}")

# Audio effects tools
@mcp.tool()
def apply_reverb(request: Optional[ReverbRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply reverb effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying reverb effect")
        processor.effects.reverb(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "reverb"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Reverb effect applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying reverb: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply reverb: {str(e)}"
        )

@mcp.tool()
def apply_delay(request: Optional[DelayRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply delay effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying delay effect")
        processor.effects.delay(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "delay"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Delay effect applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying delay: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply delay: {str(e)}"
        )

@mcp.tool()
def apply_chorus(request: Optional[ChorusRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply chorus effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying chorus effect")
        processor.effects.chorus(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "chorus"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Chorus effect applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying chorus: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply chorus: {str(e)}"
        )

@mcp.tool()
def apply_phaser(request: Optional[PhaserRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply phaser effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying phaser effect")
        processor.effects.phaser(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "phaser"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Phaser effect applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying phaser: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply phaser: {str(e)}"
        )

@mcp.tool()
def apply_compressor(request: Optional[CompressorRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply compressor effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying compressor effect")
        processor.effects.compressor(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "compressor"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Compressor effect applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying compressor: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply compressor: {str(e)}"
        )

@mcp.tool()
def apply_distortion(request: Optional[DistortionRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply distortion effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying distortion effect")
        processor.effects.distortion(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "distortion"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Distortion effect applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying distortion: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply distortion: {str(e)}"
        )

@mcp.tool()
def apply_gain(request: Optional[GainRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply gain effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying gain effect")
        processor.effects.gain(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "gain"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Gain effect applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying gain: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply gain: {str(e)}"
        )

@mcp.tool()
def apply_effect_pitch_shift(request: Optional[EffectPitchShiftRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply pitch shift effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying pitch shift effect")
        processor.effects.pitch_shift(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "pitch_shift"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Pitch shift effect applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying pitch shift: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply pitch shift: {str(e)}"
        )

@mcp.tool()
def apply_lowpass_filter(request: Optional[FilterRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply low-pass filter to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying low-pass filter")
        processor.effects.lowpass_filter(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "lowpass_filter"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Low-pass filter applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying low-pass filter: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply low-pass filter: {str(e)}"
        )

@mcp.tool()
def apply_highpass_filter(request: Optional[FilterRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply high-pass filter to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying high-pass filter")
        processor.effects.highpass_filter(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "highpass_filter"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="High-pass filter applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying high-pass filter: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply high-pass filter: {str(e)}"
        )

@mcp.tool()
def apply_noise_gate(request: Optional[NoiseGateRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply noise gate to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying noise gate")
        processor.effects.noise_gate(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "noise_gate"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Noise gate applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying noise gate: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply noise gate: {str(e)}"
        )

@mcp.tool()
def apply_limiter(request: Optional[LimiterRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Apply limiter to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying limiter")
        processor.effects.limiter(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "limiter"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Limiter applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying limiter: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply limiter: {str(e)}"
        )

@mcp.tool()
def apply_effect_chain(request: EffectChainRequest, ctx: Context = None) -> AudioProcessingResponse:
    """Apply a chain of effects to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info(f"Applying effect chain with {len(request.effect_chain)} effects")
        
        # Convert the effect_chain list into a ChainRequest
        chain_request = ChainRequest(effects=request.effect_chain)
        processor.effects.chain(chain_request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "effect_chain"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message=f"Effect chain with {len(request.effect_chain)} effects applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying effect chain: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply effect chain: {str(e)}"
        )

# Audio transformation tools
@mcp.tool()
def transform_time_stretch(request: Optional[TimeStretchRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Stretch or compress audio in time without affecting pitch."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying time stretch transformation")
        processor.transform.time_stretch(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "time_stretch"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Time stretch transformation applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying time stretch: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply time stretch: {str(e)}"
        )

@mcp.tool()
def transform_pitch_shift(request: Optional[TransformPitchShiftRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Shift the pitch of audio without affecting duration."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying pitch shift transformation")
        processor.transform.pitch_shift(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "pitch_shift"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Pitch shift transformation applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying pitch shift transformation: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply pitch shift transformation: {str(e)}"
        )

@mcp.tool()
def transform_time_and_pitch(request: Optional[TimeAndPitchRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Transform audio by changing both time and pitch simultaneously."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying time and pitch transformation")
        processor.transform.time_and_pitch(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "time_and_pitch"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Time and pitch transformation applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying time and pitch transformation: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply time and pitch transformation: {str(e)}"
        )

@mcp.tool()
def transform_resample(request: Optional[ResampleRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Resample audio to a different sample rate."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info(f"Resampling audio to {request.target_sample_rate if request else 'default'} Hz")
        processor.transform.resample(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "resample"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message=f"Audio resampled to {processor.sample_rate} Hz successfully"
        )
    except Exception as e:
        logger.error(f"Error resampling audio: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to resample audio: {str(e)}"
        )

@mcp.tool()
def transform_normalize(target_db: float = -1.0, ctx: Context = None) -> AudioProcessingResponse:
    """Normalize the audio level to a target peak level."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info(f"Normalizing audio to {target_db} dB")
        processor.transform.normalize(target_db=target_db)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "normalize"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message=f"Audio normalized to {target_db} dB successfully"
        )
    except Exception as e:
        logger.error(f"Error normalizing audio: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to normalize audio: {str(e)}"
        )

@mcp.tool()
def transform_reverse(ctx: Context = None) -> AudioProcessingResponse:
    """Reverse the audio in time."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Reversing audio")
        processor.transform.reverse()
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "reverse"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Audio reversed successfully"
        )
    except Exception as e:
        logger.error(f"Error reversing audio: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to reverse audio: {str(e)}"
        )

@mcp.tool()
def transform_fade_in(duration_seconds: float = 1.0, ctx: Context = None) -> AudioProcessingResponse:
    """Apply a fade-in effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info(f"Applying {duration_seconds}s fade-in")
        processor.transform.fade_in(duration_seconds=duration_seconds)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "fade_in"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message=f"Fade-in of {duration_seconds}s applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying fade-in: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply fade-in: {str(e)}"
        )

@mcp.tool()
def transform_fade_out(duration_seconds: float = 1.0, ctx: Context = None) -> AudioProcessingResponse:
    """Apply a fade-out effect to the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info(f"Applying {duration_seconds}s fade-out")
        processor.transform.fade_out(duration_seconds=duration_seconds)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "fade_out"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message=f"Fade-out of {duration_seconds}s applied successfully"
        )
    except Exception as e:
        logger.error(f"Error applying fade-out: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply fade-out: {str(e)}"
        )

@mcp.tool()
def transform_trim_silence(
    threshold_db: float = -60.0, 
    pad_seconds: float = 0.1,
    ctx: Context = None
) -> AudioProcessingResponse:
    """Trim silence from the beginning and end of the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info(f"Trimming silence with threshold {threshold_db} dB")
        processor.transform.trim_silence(threshold_db=threshold_db, pad_seconds=pad_seconds)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"transform": "trim_silence"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Silence trimmed successfully"
        )
    except Exception as e:
        logger.error(f"Error trimming silence: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to trim silence: {str(e)}"
        )

# Audio source separation tools
@mcp.tool()
def separate_stems(request: Optional[SeparationRequest] = None, ctx: Context = None) -> SeparationRegistryResponse:
    """Separate audio into multiple stems (vocals, drums, bass, etc.)."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Separating audio into stems")
        stems = processor.separate.separate_stems(request)
        
        # Register each stem in the registry and save to file
        registry = ctx.request_context.lifespan_context.audio_registry
        stem_ids = {}
        
        for name, audio in stems.items():
            # Register in registry
            stem_id = registry.register(
                audio_data=audio,
                sample_rate=processor.sample_rate,
                name=name,
                metadata={"type": "stem", "stem_name": name}
            )
            stem_ids[name] = stem_id
            
            # Save to file
            output_path = ctx.request_context.lifespan_context.output_dir / f"{name}.wav"
            stem_processor = AudioProcessor(audio_data=audio, sample_rate=processor.sample_rate)
            stem_processor.save(str(output_path))
        
        return SeparationRegistryResponse(
            stem_ids=stem_ids,
            sample_rate=processor.sample_rate
        )
    except Exception as e:
        logger.error(f"Error separating stems: {str(e)}")
        raise ValueError(f"Failed to separate stems: {str(e)}")

@mcp.tool()
def separate_vocals(request: Optional[TwoStemSeparationRequest] = None, ctx: Context = None) -> VocalsRegistryResponse:
    """Separate vocals from the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Separating vocals")
        vocals, accompaniment = processor.separate.vocals(request)
        
        # Register both stems in the registry
        registry = ctx.request_context.lifespan_context.audio_registry
        vocals_id = registry.register(
            audio_data=vocals,
            sample_rate=processor.sample_rate,
            name="vocals",
            metadata={"type": "stem", "content": "vocals"}
        )
        
        accompaniment_id = registry.register(
            audio_data=accompaniment,
            sample_rate=processor.sample_rate,
            name="accompaniment",
            metadata={"type": "stem", "content": "accompaniment"}
        )
        
        # Also save to files for convenience
        output_dir = ctx.request_context.lifespan_context.output_dir
        vocals_path = output_dir / "vocals.wav"
        accompaniment_path = output_dir / "accompaniment.wav"
        
        vocals_processor = AudioProcessor(audio_data=vocals, sample_rate=processor.sample_rate)
        vocals_processor.save(str(vocals_path))
        
        accompaniment_processor = AudioProcessor(audio_data=accompaniment, sample_rate=processor.sample_rate)
        accompaniment_processor.save(str(accompaniment_path))
        
        return VocalsRegistryResponse(
            vocals_id=vocals_id,
            vocals_path=str(vocals_path),
            accompaniment_id=accompaniment_id,
            accompaniment_path=str(accompaniment_path),
            sample_rate=processor.sample_rate
        )
    except Exception as e:
        logger.error(f"Error separating vocals: {str(e)}")
        raise ValueError(f"Failed to separate vocals: {str(e)}")

@mcp.tool()
def separate_drums(request: Optional[TwoStemSeparationRequest] = None, ctx: Context = None) -> TwoStemRegistryResponse:
    """Separate drums from the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Separating drums")
        if request is None:
            # Create a request with stem set to drums
            request = TwoStemSeparationRequest(stem="drums")
        else:
            # Make sure the stem is set to drums
            request.stem = "drums"
            
        drums, other = processor.separate.two_stems(request)
        
        # Register both stems in the registry
        registry = ctx.request_context.lifespan_context.audio_registry
        drums_id = registry.register(
            audio_data=drums,
            sample_rate=processor.sample_rate,
            name="drums",
            metadata={"type": "stem", "content": "drums"}
        )
        
        other_id = registry.register(
            audio_data=other,
            sample_rate=processor.sample_rate,
            name="other",
            metadata={"type": "stem", "content": "other"}
        )
        
        # Also save to files for convenience
        output_dir = ctx.request_context.lifespan_context.output_dir
        drums_path = output_dir / "drums.wav"
        other_path = output_dir / "other.wav"
        
        drums_processor = AudioProcessor(audio_data=drums, sample_rate=processor.sample_rate)
        drums_processor.save(str(drums_path))
        
        other_processor = AudioProcessor(audio_data=other, sample_rate=processor.sample_rate)
        other_processor.save(str(other_path))
        
        return TwoStemRegistryResponse(
            target_stem_id=drums_id,
            target_stem_path=str(drums_path),
            remainder_id=other_id,
            remainder_path=str(other_path),
            sample_rate=processor.sample_rate
        )
    except Exception as e:
        logger.error(f"Error separating drums: {str(e)}")
        raise ValueError(f"Failed to separate drums: {str(e)}")

@mcp.tool()
def separate_bass(request: Optional[TwoStemSeparationRequest] = None, ctx: Context = None) -> TwoStemRegistryResponse:
    """Separate bass from the audio."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Separating bass")
        if request is None:
            # Create a request with stem set to bass
            request = TwoStemSeparationRequest(stem="bass")
        else:
            # Make sure the stem is set to bass
            request.stem = "bass"
            
        bass, other = processor.separate.two_stems(request)
        
        # Register both stems in the registry
        registry = ctx.request_context.lifespan_context.audio_registry
        bass_id = registry.register(
            audio_data=bass,
            sample_rate=processor.sample_rate,
            name="bass",
            metadata={"type": "stem", "content": "bass"}
        )
        
        other_id = registry.register(
            audio_data=other,
            sample_rate=processor.sample_rate,
            name="other",
            metadata={"type": "stem", "content": "other"}
        )
        
        # Also save to files for convenience
        output_dir = ctx.request_context.lifespan_context.output_dir
        bass_path = output_dir / "bass.wav"
        other_path = output_dir / "other.wav"
        
        bass_processor = AudioProcessor(audio_data=bass, sample_rate=processor.sample_rate)
        bass_processor.save(str(bass_path))
        
        other_processor = AudioProcessor(audio_data=other, sample_rate=processor.sample_rate)
        other_processor.save(str(other_path))
        
        return TwoStemRegistryResponse(
            target_stem_id=bass_id,
            target_stem_path=str(bass_path),
            remainder_id=other_id,
            remainder_path=str(other_path),
            sample_rate=processor.sample_rate
        )
    except Exception as e:
        logger.error(f"Error separating bass: {str(e)}")
        raise ValueError(f"Failed to separate bass: {str(e)}")

@mcp.tool()
def apply_karaoke(request: Optional[TwoStemSeparationRequest] = None, ctx: Context = None) -> AudioProcessingResponse:
    """Remove vocals from the audio (karaoke effect)."""
    try:
        processor = ctx.request_context.lifespan_context.processor
        if processor is None or processor.audio_data is None:
            raise ValueError("No audio data loaded")
        
        logger.info("Applying karaoke effect (vocal removal)")
        processor.separate.apply_vocal_removal(request)
        
        # Update the current audio in registry
        registry = ctx.request_context.lifespan_context.audio_registry
        registry.register(
            audio_data=processor.audio_data,
            sample_rate=processor.sample_rate,
            name="current",
            metadata={"effect": "karaoke"}
        )
        
        return AudioProcessingResponse(
            success=True,
            message="Vocals removed successfully (karaoke effect)"
        )
    except Exception as e:
        logger.error(f"Error applying karaoke effect: {str(e)}")
        return AudioProcessingResponse(
            success=False,
            message=f"Failed to apply karaoke effect: {str(e)}"
        )

def main():
    mcp.run()
    
if __name__ == "__main__":
    main()