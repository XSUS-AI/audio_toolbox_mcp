# Audio Registry Implementation

## Overview

The Audio Registry is a management system for in-memory audio data in the Audio Toolbox MCP server. It allows agents to work with audio processing operations without dealing with unwieldy base64 encoded data.

## Key Components

### 1. AudioRegistry Class

```python
class AudioRegistry:
    """Registry for storing audio data with access via IDs."""
    
    def __init__(self):
        self.audio_registry = {}  # Maps IDs to (audio_data, sample_rate, metadata)
        self.counter = 0  # For generating unique IDs
```

The registry maintains an internal dictionary of audio entries, each containing:
- Audio data (numpy array)
- Sample rate
- Metadata (source information, processing history, etc.)
- Creation timestamp

### 2. Registry Operations

- `register(audio_data, sample_rate, name=None, metadata=None)`: Add or update audio data in the registry
- `get(audio_id)`: Retrieve audio data by ID
- `list_entries()`: Get metadata for all entries
- `remove(audio_id)`: Delete audio entry
- `clear()`: Empty the registry

## Registry Integration

### 1. Application Context

The registry is stored in the application lifespan context:

```python
class AppContext:
    def __init__(self):
        self.processor = None
        self.audio_registry = AudioRegistry()
        self.output_dir = Path("output")
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
```

### 2. Registry Management Tools

Tools for working with the registry:

- `list_audio_registry`: View all audio entries in the registry
- `load_audio_from_registry`: Load audio from registry into processor
- `remove_from_audio_registry`: Delete an entry
- `clear_audio_registry`: Remove all entries

### 3. Audio Processing Flow

1. **Loading Audio**:
   - When audio is loaded from a file or data, it's automatically registered with ID "current"
   - Example: `load_audio_file({"file_path": "sample.mp3"})`

2. **Processing Audio**:
   - All processing operations (effects, transformations) update the "current" entry
   - Example: `apply_reverb({"room_size": 0.8})`

3. **Audio Separation**:
   - When audio is separated, each component is registered with a unique ID
   - Files are also saved to the output directory
   - Example: `separate_vocals()` returns:
     ```json
     {
       "vocals_id": "vocals",
       "vocals_path": "output/vocals.wav",
       "accompaniment_id": "accompaniment",
       "accompaniment_path": "output/accompaniment.wav",
       "sample_rate": 44100
     }
     ```

4. **Working with Registry Entries**:
   - Load an entry to make it the "current" audio:
     ```python
     load_audio_from_registry({"audio_id": "vocals"})
     ```
   - Apply effects to it:
     ```python
     apply_reverb({"room_size": 0.8})
     ```
   - Save the result:
     ```python
     save_audio({"file_path": "output/vocals_reverb.wav"})
     ```

## Benefits for Agents

1. **Token Efficiency**: No large base64 strings in agent context
2. **Intuitive References**: Audio referred to by descriptive IDs
3. **Automatic File Saving**: All processed audio is automatically saved to files
4. **State Management**: Registry maintains history of processed audio

## Example Workflow

```
User: Load the audio file "guitar.mp3" and separate the vocals from the background music.

Agent: 
[load_audio_file -> AudioInfoResponse with duration, sample_rate, num_channels]
[separate_vocals -> VocalsRegistryResponse with vocals_id, vocals_path, accompaniment_id, accompaniment_path]

The vocals have been separated and saved to output/vocals.wav and output/accompaniment.wav.

User: Apply some reverb to the vocals and save it as "vocals_reverb.wav"

Agent:
[load_audio_from_registry with audio_id="vocals" -> AudioInfoResponse]
[apply_reverb with room_size=0.8 -> AudioProcessingResponse]
[save_audio with file_path="output/vocals_reverb.wav" -> AudioProcessingResponse]

I've applied reverb to the vocals and saved the result to output/vocals_reverb.wav.
```
