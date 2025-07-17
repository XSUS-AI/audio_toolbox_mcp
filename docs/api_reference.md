# Audio Toolbox API Reference

This document provides a comprehensive reference for the Audio Toolbox API.

## Table of Contents

- [AudioProcessor](#audioprocessor)
- [AudioAnalyzer](#audioanalyzer)
- [AudioEffects](#audioeffects)
- [AudioSeparator](#audioseparator)
- [AudioTransformer](#audiotransformer)

## AudioProcessor

The main class for audio processing, providing access to all audio toolbox functionality.

### Constructor

```python
AudioProcessor(file_path=None, audio_data=None, sample_rate=None)
```

**Parameters:**
- `file_path` (str, Path, optional): Path to an audio file to load
- `audio_data` (numpy.ndarray, optional): NumPy array containing audio data
- `sample_rate` (int, optional): Sample rate of the audio data (required if audio_data is provided)

### Properties

- `audio_data` (numpy.ndarray): The current audio data
- `sample_rate` (int): The current sample rate
- `duration` (float): Duration of the audio in seconds
- `num_channels` (int): Number of audio channels
- `analyze` (AudioAnalyzer): Interface to audio analysis features
- `effects` (AudioEffects): Interface to audio effects
- `separate` (AudioSeparator): Interface to audio source separation
- `transform` (AudioTransformer): Interface to audio transformations

### Methods

#### load_file

```python
load_file(file_path)
```

Load audio from a file

**Parameters:**
- `file_path` (str or Path): Path to audio file to load

#### save

```python
save(file_path, format=None)
```

Save the current audio data to a file

**Parameters:**
- `file_path` (str or Path): Path to save audio to
- `format` (str, optional): Format to save audio as (inferred from file extension if None)

#### reset

```python
reset()
```

Clear all audio data

## AudioAnalyzer

Audio analysis functionality using librosa, audioFlux, and pyAudioAnalysis.

### Methods

#### mel_spectrogram

```python
mel_spectrogram(request=None)
```

Calculate the mel spectrogram of the audio.

**Parameters:**
- `request` (MelSpectrogramRequest, optional): Parameters for mel spectrogram calculation

**Returns:**
- Mel spectrogram as a numpy array

#### chromagram

```python
chromagram(request=None)
```

Calculate the chromagram of the audio.

**Parameters:**
- `request` (ChromagramRequest, optional): Parameters for chromagram calculation

**Returns:**
- Chromagram as a numpy array

#### detect_onsets

```python
detect_onsets(request=None)
```

Detect onsets in the audio signal.

**Parameters:**
- `request` (OnsetDetectionRequest, optional): Parameters for onset detection

**Returns:**
- Array of onset times in seconds

#### track_beats

```python
track_beats(request=None)
```

Track beats in the audio signal.

**Parameters:**
- `request` (BeatTrackingRequest, optional): Parameters for beat tracking

**Returns:**
- Tuple containing (beat_times, tempo)

#### spectral_features

```python
spectral_features(request=None)
```

Extract spectral features from the audio signal.

**Parameters:**
- `request` (SpectralFeaturesRequest, optional): Parameters for spectral feature extraction

**Returns:**
- Dictionary of spectral features

#### track_pitch

```python
track_pitch(request=None)
```

Track pitch in the audio signal.

**Parameters:**
- `request` (PitchTrackingRequest, optional): Parameters for pitch tracking

**Returns:**
- Tuple containing (times, frequencies)

#### visualize_spectrogram

```python
visualize_spectrogram(log_scale=True)
```

Visualize the spectrogram of the audio.

**Parameters:**
- `log_scale` (bool): Whether to use log scale for the spectrogram

**Returns:**
- Matplotlib figure containing the spectrogram visualization

#### visualize_waveform

```python
visualize_waveform()
```

Visualize the waveform of the audio.

**Returns:**
- Matplotlib figure containing the waveform visualization

## AudioEffects

Audio effects processing using pedalboard library.

### Methods

#### reverb

```python
reverb(request=None)
```

Apply reverb effect to the audio.

**Parameters:**
- `request` (ReverbRequest, optional): Parameters for the reverb effect

**Returns:**
- The AudioProcessor instance with processed audio

#### delay

```python
delay(request=None)
```

Apply delay effect to the audio.

**Parameters:**
- `request` (DelayRequest, optional): Parameters for the delay effect

**Returns:**
- The AudioProcessor instance with processed audio

#### chorus

```python
chorus(request=None)
```

Apply chorus effect to the audio.

**Parameters:**
- `request` (ChorusRequest, optional): Parameters for the chorus effect

**Returns:**
- The AudioProcessor instance with processed audio

#### phaser

```python
phaser(request=None)
```

Apply phaser effect to the audio.

**Parameters:**
- `request` (PhaserRequest, optional): Parameters for the phaser effect

**Returns:**
- The AudioProcessor instance with processed audio

#### compressor

```python
compressor(request=None)
```

Apply compressor effect to the audio.

**Parameters:**
- `request` (CompressorRequest, optional): Parameters for the compressor effect

**Returns:**
- The AudioProcessor instance with processed audio

#### distortion

```python
distortion(request=None)
```

Apply distortion effect to the audio.

**Parameters:**
- `request` (DistortionRequest, optional): Parameters for the distortion effect

**Returns:**
- The AudioProcessor instance with processed audio

#### gain

```python
gain(request=None)
```

Apply gain effect to the audio.

**Parameters:**
- `request` (GainRequest, optional): Parameters for the gain effect

**Returns:**
- The AudioProcessor instance with processed audio

#### pitch_shift

```python
pitch_shift(request=None)
```

Apply pitch shift effect to the audio.

**Parameters:**
- `request` (PitchShiftRequest, optional): Parameters for the pitch shift effect

**Returns:**
- The AudioProcessor instance with processed audio

#### lowpass_filter

```python
lowpass_filter(request=None)
```

Apply low-pass filter effect to the audio.

**Parameters:**
- `request` (FilterRequest, optional): Parameters for the filter effect

**Returns:**
- The AudioProcessor instance with processed audio

#### highpass_filter

```python
highpass_filter(request=None)
```

Apply high-pass filter effect to the audio.

**Parameters:**
- `request` (FilterRequest, optional): Parameters for the filter effect

**Returns:**
- The AudioProcessor instance with processed audio

#### noise_gate

```python
noise_gate(request=None)
```

Apply noise gate effect to the audio.

**Parameters:**
- `request` (NoiseGateRequest, optional): Parameters for the noise gate effect

**Returns:**
- The AudioProcessor instance with processed audio

#### limiter

```python
limiter(request=None)
```

Apply limiter effect to the audio.

**Parameters:**
- `request` (LimiterRequest, optional): Parameters for the limiter effect

**Returns:**
- The AudioProcessor instance with processed audio

#### chain

```python
chain(request)
```

Apply a chain of effects to the audio.

**Parameters:**
- `request` (ChainRequest): Parameters for the effect chain

**Returns:**
- The AudioProcessor instance with processed audio

## AudioSeparator

Audio source separation functionality using Demucs.

### Methods

#### separate_stems

```python
separate_stems(request=None)
```

Separate audio into multiple stems.

**Parameters:**
- `request` (SeparationRequest, optional): Parameters for audio separation

**Returns:**
- Dictionary of separated stems (vocals, drums, bass, other, etc.)

#### vocals

```python
vocals(request=None)
```

Separate vocals from audio.

**Parameters:**
- `request` (TwoStemSeparationRequest, optional): Parameters for vocal separation

**Returns:**
- Tuple containing (vocals, accompaniment)

#### drums

```python
drums(request=None)
```

Separate drums from audio.

**Parameters:**
- `request` (TwoStemSeparationRequest, optional): Parameters for drum separation

**Returns:**
- Tuple containing (drums, accompaniment without drums)

#### bass

```python
bass(request=None)
```

Separate bass from audio.

**Parameters:**
- `request` (TwoStemSeparationRequest, optional): Parameters for bass separation

**Returns:**
- Tuple containing (bass, accompaniment without bass)

#### two_stems

```python
two_stems(request=None)
```

Separate audio into two stems: target stem and everything else.

**Parameters:**
- `request` (TwoStemSeparationRequest, optional): Parameters for two-stem separation

**Returns:**
- Tuple containing (target_stem, everything_else)

#### apply_separation

```python
apply_separation(stem_name, request=None)
```

Apply separation and update the processor's audio with the specified stem.

**Parameters:**
- `stem_name` (str): Name of the stem to keep ("vocals", "drums", "bass", etc.)
- `request` (TwoStemSeparationRequest, optional): Parameters for separation

**Returns:**
- The AudioProcessor instance with processed audio

#### apply_vocal_removal

```python
apply_vocal_removal(request=None)
```

Remove vocals from the audio (karaoke effect).

**Parameters:**
- `request` (TwoStemSeparationRequest, optional): Parameters for vocal separation

**Returns:**
- The AudioProcessor instance with processed audio

#### available_models

```python
available_models()
```

Get a list of available Demucs models.

**Returns:**
- List of available model names

## AudioTransformer

Audio transformation functionality combining time and frequency domain transformations.

### Methods

#### time_stretch

```python
time_stretch(request=None)
```

Stretch or compress audio in time without affecting pitch.

**Parameters:**
- `request` (TimeStretchRequest, optional): Parameters for time stretching

**Returns:**
- The AudioProcessor instance with transformed audio

#### pitch_shift

```python
pitch_shift(request=None)
```

Shift the pitch of audio without affecting its duration.

**Parameters:**
- `request` (PitchShiftRequest, optional): Parameters for pitch shifting

**Returns:**
- The AudioProcessor instance with transformed audio

#### time_and_pitch

```python
time_and_pitch(request=None)
```

Transform audio by changing both time and pitch simultaneously.

**Parameters:**
- `request` (TimeAndPitchRequest, optional): Parameters for time and pitch transformation

**Returns:**
- The AudioProcessor instance with transformed audio

#### resample

```python
resample(request=None)
```

Resample audio to a different sample rate.

**Parameters:**
- `request` (ResampleRequest, optional): Parameters for resampling

**Returns:**
- The AudioProcessor instance with resampled audio

#### normalize

```python
normalize(target_db=-1.0)
```

Normalize the audio to have a specific peak level.

**Parameters:**
- `target_db` (float): Target peak level in dB (0 = maximum digital value)

**Returns:**
- The AudioProcessor instance with normalized audio

#### reverse

```python
reverse()
```

Reverse the audio in time.

**Returns:**
- The AudioProcessor instance with reversed audio

#### fade_in

```python
fade_in(duration_seconds=1.0)
```

Apply a fade-in effect to the audio.

**Parameters:**
- `duration_seconds` (float): Duration of the fade-in in seconds

**Returns:**
- The AudioProcessor instance with fade-in applied

#### fade_out

```python
fade_out(duration_seconds=1.0)
```

Apply a fade-out effect to the audio.

**Parameters:**
- `duration_seconds` (float): Duration of the fade-out in seconds

**Returns:**
- The AudioProcessor instance with fade-out applied

#### trim_silence

```python
trim_silence(threshold_db=-60.0, pad_seconds=0.1)
```

Trim silence from the beginning and end of the audio.

**Parameters:**
- `threshold_db` (float): Threshold for silence detection in dB
- `pad_seconds` (float): Seconds of padding to leave around non-silent regions

**Returns:**
- The AudioProcessor instance with trimmed audio
