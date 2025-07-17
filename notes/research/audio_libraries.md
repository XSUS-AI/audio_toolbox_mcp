# Audio Analysis and Transformation Libraries Research

## Main Audio Processing Libraries

### 1. Librosa
**Description**: A Python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

**Key Features**:
- Loading and visualizing audio files
- Spectral analysis (mel spectrogram, chromagram, etc.)
- Feature extraction (MFCCs, spectral contrast, chroma, zero crossing rate, etc.)
- Beat detection and tempo estimation
- Pitch tracking
- Onset detection
- Segmentation

**Strengths**: Comprehensive feature extraction, well-documented, widely used in research

**Usage**: Primarily for analysis rather than transformation

**Source**: [GitHub - Librosa](https://librosa.org/doc/)

### 2. pyAudioAnalysis
**Description**: Python library for audio feature extraction, classification, segmentation and applications

**Key Features**:
- Audio feature extraction (time and frequency domain features)
- Audio classification 
- Audio segmentation (supervised and unsupervised)
- Content-based audio retrieval and audio thumbnailing
- Audio event detection
- Speaker diarization (who spoke when)

**Strengths**: Good for classification tasks and machine learning applications

**Source**: [GitHub - pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)

### 3. audioFlux
**Description**: A deep learning tool library for audio and music analysis, feature extraction.

**Key Features**:
- Multiple time-frequency transformations (BFT, NSGT, CWT, PWT, CQT, VQT, etc.)
- Various frequency scale types (Linear, Mel, Bark, Erb, Octave, Log)
- Spectral feature extraction
- Deconvolution for spectrum
- Chroma feature extraction
- Pitch detection (multiple algorithms including YIN, STFT, etc.)
- Onset detection
- Harmonic-percussive source separation (HPSS)
- Pitch shifting and time stretching algorithms

**Strengths**: Comprehensive transformation capabilities, good for deep learning use cases

**Source**: [GitHub - audioFlux](https://github.com/libAudioFlux/audioFlux)

### 4. Pedalboard (Spotify)
**Description**: A Python library for audio processing, with a focus on applying audio effects like a guitar pedalboard.

**Key Features**:
- Built-in audio I/O utilities (reading/writing various audio formats)
- Guitar-style effects: Chorus, Distortion, Phaser, Clipping
- Loudness and dynamic range effects: Compressor, Gain, Limiter
- Equalizers and filters: HighpassFilter, LadderFilter, LowpassFilter
- Spatial effects: Convolution, Delay, Reverb
- Pitch effects: PitchShift
- Lossy compression: GSMFullRateCompressor, MP3Compressor
- Quality reduction: Resample, Bitcrush
- Support for VST3 and Audio Unit plugins
- Thread-safety and speed optimizations

**Strengths**: Production-ready audio effects, excellent performance, VST plugin support

**Source**: [GitHub - Pedalboard](https://github.com/spotify/pedalboard)

### 5. Demucs (Facebook Research)
**Description**: A state-of-the-art music source separation model for isolating vocal, bass, drums, and other components from music tracks.

**Key Features**:
- High-quality audio source separation
- Multiple pre-trained models
- Hybrid Transformer architecture
- Supports separation into drums, bass, vocals, and other stems
- Experimental 6-source model with guitar and piano isolation
- Options for quality vs. memory usage tradeoffs

**Strengths**: Best-in-class for isolating audio components from mixed tracks

**Source**: [GitHub - Demucs](https://github.com/facebookresearch/demucs)

## Additional Audio Processing Libraries

### 1. pydub
- Audio file manipulation (cutting, concatenating, crossfading)
- Basic effects (gain adjustment, normalization, etc.)
- Simple API for converting between formats

### 2. SoundFile
- Reading and writing sound files
- Low-level audio file manipulation

### 3. PyAudio
- Recording and playing audio
- Real-time audio processing

## Comparison for Our Needs

For a comprehensive audio toolbox that can handle analysis and transformations:

1. **Core Analysis**: Librosa provides excellent tools for feature extraction, spectral analysis, and general audio analysis.

2. **Audio Effects**: Pedalboard offers the most comprehensive set of audio effects and transformations with excellent performance.

3. **Source Separation**: Demucs offers state-of-the-art capabilities for separating audio sources (vocals, drums, bass, etc.)

4. **Advanced Transformations**: audioFlux provides additional transformation capabilities particularly useful for advanced audio manipulation.

5. **Classification**: pyAudioAnalysis offers good tools for audio classification and segmentation.

A combination of these libraries would provide a comprehensive toolkit for audio analysis and transformation, covering everything from basic effects to advanced source separation and deep learning features.