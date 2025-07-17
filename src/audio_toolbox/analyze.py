import numpy as np
import librosa
import audioflux as af
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Dict, Any, List, TYPE_CHECKING
from pydantic import BaseModel, Field

# Type checking imports
if TYPE_CHECKING:
    from audio_toolbox.processor import AudioProcessor

class MelSpectrogramRequest(BaseModel):
    """Parameters for mel spectrogram generation"""
    n_fft: int = Field(2048, description="FFT window size")
    hop_length: int = Field(512, description="Number of samples between frames")
    n_mels: int = Field(128, description="Number of mel bands")
    fmin: float = Field(0.0, description="Minimum frequency (Hz)")
    fmax: Optional[float] = Field(None, description="Maximum frequency (Hz)")
    power: float = Field(2.0, description="Power for spectrogram normalization")

class ChromagramRequest(BaseModel):
    """Parameters for chromagram generation"""
    n_fft: int = Field(2048, description="FFT window size")
    hop_length: int = Field(512, description="Number of samples between frames")
    n_chroma: int = Field(12, description="Number of chroma bins")
    tuning: Optional[float] = Field(None, description="Tuning offset in fractions of a bin")

class OnsetDetectionRequest(BaseModel):
    """Parameters for onset detection"""
    n_fft: int = Field(2048, description="FFT window size")
    hop_length: int = Field(512, description="Number of samples between frames")
    backtrack: bool = Field(False, description="Whether to backtrack to nearest minimum before onset")
    pre_max: int = Field(30, description="Pre-onset max window size in samples")
    post_max: int = Field(30, description="Post-onset max window size in samples")
    pre_avg: int = Field(100, description="Pre-onset average window size in samples")
    post_avg: int = Field(100, description="Post-onset average window size in samples")
    delta: float = Field(0.07, description="Threshold for onset detection")
    wait: int = Field(30, description="Number of samples to wait between onsets")

class BeatTrackingRequest(BaseModel):
    """Parameters for beat tracking"""
    start_bpm: float = Field(120.0, description="Estimated starting tempo in BPM")
    hop_length: int = Field(512, description="Number of samples between frames")

class SpectralFeaturesRequest(BaseModel):
    """Parameters for spectral features extraction"""
    n_fft: int = Field(2048, description="FFT window size")
    hop_length: int = Field(512, description="Number of samples between frames")

class PitchTrackingRequest(BaseModel):
    """Parameters for pitch tracking"""
    method: str = Field("yin", description="Method used for pitch tracking (yin, pyin, crepe, etc.)")
    fmin: float = Field(50.0, description="Minimum frequency to look for (Hz)")
    fmax: float = Field(2000.0, description="Maximum frequency to look for (Hz)")
    hop_length: int = Field(512, description="Number of samples between frames")

class AudioAnalyzer:
    """
    Audio analysis functionality using librosa, audioFlux, and pyAudioAnalysis.
    
    This class provides methods for extracting features and analyzing audio signals.
    """
    
    def __init__(self, processor: 'AudioProcessor'):
        """
        Initialize the analyzer with a reference to the AudioProcessor.
        
        Args:
            processor: The parent AudioProcessor instance
        """
        self._processor = processor
    
    def _check_audio(self):
        """Check that audio data is available for analysis"""
        if self._processor.audio_data is None or self._processor.sample_rate is None:
            raise ValueError("No audio data available for analysis")
    
    def _get_mono_audio(self):
        """Get mono audio from the processor's audio data"""
        self._check_audio()
        if self._processor.audio_data.shape[0] > 1:  # Multi-channel audio
            # Average all channels to get mono
            return np.mean(self._processor.audio_data, axis=0)
        return self._processor.audio_data[0]  # Already mono
    
    def mel_spectrogram(self, request: Optional[MelSpectrogramRequest] = None) -> np.ndarray:
        """
        Calculate the mel spectrogram of the audio.
        
        Args:
            request: Parameters for mel spectrogram calculation
            
        Returns:
            Mel spectrogram as a numpy array
        """
        if request is None:
            request = MelSpectrogramRequest()
            
        mono_audio = self._get_mono_audio()
        
        return librosa.feature.melspectrogram(
            y=mono_audio, 
            sr=self._processor.sample_rate,
            n_fft=request.n_fft,
            hop_length=request.hop_length,
            n_mels=request.n_mels,
            fmin=request.fmin,
            fmax=request.fmax,
            power=request.power
        )
    
    def chromagram(self, request: Optional[ChromagramRequest] = None) -> np.ndarray:
        """
        Calculate the chromagram of the audio.
        
        Args:
            request: Parameters for chromagram calculation
            
        Returns:
            Chromagram as a numpy array
        """
        if request is None:
            request = ChromagramRequest()
            
        mono_audio = self._get_mono_audio()
        
        return librosa.feature.chroma_stft(
            y=mono_audio,
            sr=self._processor.sample_rate,
            n_fft=request.n_fft,
            hop_length=request.hop_length,
            n_chroma=request.n_chroma,
            tuning=request.tuning
        )
    
    def detect_onsets(self, request: Optional[OnsetDetectionRequest] = None) -> np.ndarray:
        """
        Detect onsets in the audio signal.
        
        Args:
            request: Parameters for onset detection
            
        Returns:
            Array of onset times in seconds
        """
        if request is None:
            request = OnsetDetectionRequest()
            
        mono_audio = self._get_mono_audio()
        
        # Get onset frames
        onset_frames = librosa.onset.onset_detect(
            y=mono_audio,
            sr=self._processor.sample_rate,
            hop_length=request.hop_length,
            backtrack=request.backtrack,
            pre_max=request.pre_max,
            post_max=request.post_max,
            pre_avg=request.pre_avg,
            post_avg=request.post_avg,
            delta=request.delta,
            wait=request.wait
        )
        
        # Convert frames to seconds
        onset_times = librosa.frames_to_time(onset_frames, sr=self._processor.sample_rate, 
                                             hop_length=request.hop_length)
        
        return onset_times
    
    def track_beats(self, request: Optional[BeatTrackingRequest] = None) -> Tuple[np.ndarray, float]:
        """
        Track beats in the audio signal.
        
        Args:
            request: Parameters for beat tracking
            
        Returns:
            Tuple containing (beat_times, tempo)
        """
        if request is None:
            request = BeatTrackingRequest()
            
        mono_audio = self._get_mono_audio()
        
        tempo, beat_frames = librosa.beat.beat_track(
            y=mono_audio,
            sr=self._processor.sample_rate,
            start_bpm=request.start_bpm,
            hop_length=request.hop_length
        )
        
        beat_times = librosa.frames_to_time(beat_frames, sr=self._processor.sample_rate, 
                                            hop_length=request.hop_length)
        
        return beat_times, tempo
    
    def spectral_features(self, request: Optional[SpectralFeaturesRequest] = None) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from the audio signal.
        
        Args:
            request: Parameters for spectral feature extraction
            
        Returns:
            Dictionary of spectral features
        """
        if request is None:
            request = SpectralFeaturesRequest()
            
        mono_audio = self._get_mono_audio()
        
        # Calculate various spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=mono_audio, sr=self._processor.sample_rate, 
            n_fft=request.n_fft, hop_length=request.hop_length
        )
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=mono_audio, sr=self._processor.sample_rate,
            n_fft=request.n_fft, hop_length=request.hop_length
        )
        
        spectral_contrast = librosa.feature.spectral_contrast(
            y=mono_audio, sr=self._processor.sample_rate,
            n_fft=request.n_fft, hop_length=request.hop_length
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=mono_audio, sr=self._processor.sample_rate,
            n_fft=request.n_fft, hop_length=request.hop_length
        )
        
        mfccs = librosa.feature.mfcc(
            y=mono_audio, sr=self._processor.sample_rate,
            n_fft=request.n_fft, hop_length=request.hop_length, n_mfcc=13
        )
        
        return {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_contrast": spectral_contrast,
            "spectral_rolloff": spectral_rolloff,
            "mfccs": mfccs
        }
    
    def track_pitch(self, request: Optional[PitchTrackingRequest] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track pitch in the audio signal.
        
        Args:
            request: Parameters for pitch tracking
            
        Returns:
            Tuple containing (times, frequencies)
        """
        if request is None:
            request = PitchTrackingRequest()
            
        mono_audio = self._get_mono_audio()
        
        # Use the specified pitch tracking method
        if request.method == "yin":
            # Use AudioFlux YIN algorithm
            pitch_obj = af.PitchYIN(self._processor.sample_rate, request.hop_length)
            pitch_result = pitch_obj.pitch(mono_audio)
            frequencies = pitch_result['frequency']
            # Create time values
            hop_time = request.hop_length / self._processor.sample_rate
            times = np.arange(len(frequencies)) * hop_time
            
        elif request.method == "pyin":
            # Use librosa's PYIN
            frequencies, _, _ = librosa.pyin(
                mono_audio, 
                sr=self._processor.sample_rate,
                fmin=request.fmin,
                fmax=request.fmax,
                hop_length=request.hop_length
            )
            # Create time values
            hop_time = request.hop_length / self._processor.sample_rate
            times = np.arange(len(frequencies)) * hop_time
            
        else:
            # Default to YIN if method not supported
            pitch_obj = af.PitchYIN(self._processor.sample_rate, request.hop_length)
            pitch_result = pitch_obj.pitch(mono_audio)
            frequencies = pitch_result['frequency']
            # Create time values
            hop_time = request.hop_length / self._processor.sample_rate
            times = np.arange(len(frequencies)) * hop_time
        
        return times, frequencies
    
    def visualize_spectrogram(self, log_scale: bool = True) -> plt.Figure:
        """
        Visualize the spectrogram of the audio.
        
        Args:
            log_scale: Whether to use log scale for the spectrogram
            
        Returns:
            Matplotlib figure containing the spectrogram visualization
        """
        mono_audio = self._get_mono_audio()
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Calculate spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(mono_audio)), ref=np.max) if log_scale else \
            np.abs(librosa.stft(mono_audio))
        
        # Plot spectrogram
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', 
                                      sr=self._processor.sample_rate, ax=ax)
        ax.set_title('Spectrogram')
        fig.colorbar(img, ax=ax, format='%+2.0f dB' if log_scale else '%+2.0f')
        
        return fig
    
    def visualize_waveform(self) -> plt.Figure:
        """
        Visualize the waveform of the audio.
        
        Returns:
            Matplotlib figure containing the waveform visualization
        """
        self._check_audio()
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot each channel in a different color
        for i in range(self._processor.audio_data.shape[0]):
            ax.plot(np.linspace(0, self._processor.duration, len(self._processor.audio_data[i])), 
                    self._processor.audio_data[i], 
                    label=f'Channel {i+1}', alpha=0.7)
        
        ax.set_title('Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        if self._processor.audio_data.shape[0] > 1:  # Only add legend for multi-channel audio
            ax.legend()
        
        return fig
