# Audio Toolbox

A comprehensive Python library for audio analysis and transformations, combining the best open source tools.

## Features

- **Audio Analysis**: Extract features, analyze spectrograms, detect beats and onsets
- **Audio Effects**: Apply professional audio effects (reverb, delay, compression, EQ, etc.)
- **Source Separation**: Isolate vocals, drums, bass, and other components
- **Audio Transformations**: Pitch shift, time stretch, resample
- **AI Integration**: Use the AudioProcessor agent to run complex audio tasks
- **Audio Registry**: Efficient in-memory management of audio assets with automatic file saving

## Installation

### Prerequisites

- Python 3.8+
- Required libraries listed in `requirements.txt`

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/audio_toolbox.git
cd audio_toolbox

# Install dependencies
pip install -r requirements.txt
```

## Using the API

```python
from audio_toolbox import AudioProcessor

# Load an audio file
processor = AudioProcessor('path/to/audio.wav')

# Get audio features
melspec = processor.analyze.mel_spectrogram()

# Apply an effect
processed_audio = processor.effects.reverb(room_size=0.8, damping=0.5)

# Separate vocals from the track
vocals, accompaniment = processor.separate.vocals()

# Save processed audio
processor.save('output.wav')
```

## Using the AudioProcessor Agent

The AudioProcessor agent provides a conversational interface for working with audio files. It allows you to perform complex audio operations through natural language commands.

### Setup

1. Set up the required environment variables:

```bash
# Create a .env file with your OpenRouter API key
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

2. Run the agent:

```bash
python agent.py
```

3. Interact with the agent using natural language:

```
> Load my audio file sample.mp3 and analyze its tempo
```

### Example Capabilities

- Load and analyze audio files
- Apply audio effects and transformations
- Separate audio into stems (vocals, drums, bass, etc.)
- Generate visualizations
- Chain multiple operations together

Check out `examples/test_prompts.md` for example interactions with the agent.

## MCP Server

This project includes an MCP (Model Context Protocol) server that enables the AudioProcessor agent to interface with the audio processing functionality.

To run the MCP server directly:

```bash
python mcp_server.py
```

This server exposes all the audio toolbox functionality as tools that can be used by any MCP-compatible client.

### Audio Registry

The MCP server includes an Audio Registry system that efficiently manages audio data in memory using IDs instead of passing large base64 encoded strings. Key benefits:

- **Token Efficiency**: Small IDs instead of large encoded strings
- **Intuitive References**: Descriptive IDs for audio assets (e.g., "vocals", "drums")
- **Automatic File Saving**: All processed audio components are saved to files
- **State Management**: Easy tracking of audio processing history

See `docs/audio_registry.md` for complete details on using the registry system.

## Examples

Several example scripts are provided in the `examples` directory:

- `basic_example.py`: Simple demonstration of core audio toolbox functionality
- `effects_example.py`: Comprehensive examples of audio effects processing
- `source_separation_example.py`: Examples of audio source separation
- `transformation_example.py`: Examples of audio transformations
- `test_prompts.md`: Example prompts for interacting with the AudioProcessor agent

## Project Structure

```
├── agents/                      # Agent system prompts
├── docs/                        # Documentation
│   └── audio_registry.md        # Audio registry documentation
├── examples/                    # Example scripts
├── logs/                        # Log files
├── output/                      # Output directory for processed audio
├── src/                         # Source code
│   └── audio_toolbox/           # Main package
│       ├── __init__.py          # Package initialization
│       ├── processor.py         # Main AudioProcessor class
│       ├── analyze.py           # Audio analysis functionality
│       ├── effects.py           # Audio effects processing
│       ├── separate.py          # Audio source separation
│       └── transform.py         # Audio transformations
├── tests/                       # Unit tests
├── .well-known/                 # Agent card for A2A protocol
├── agent.py                     # AudioProcessor agent
├── mcp_server.py                # MCP server implementation
├── requirements.txt             # Package dependencies
└── setup.py                     # Package setup script
```

## Credits

This project combines several powerful open-source audio libraries:

- **Librosa**: Audio analysis and feature extraction
- **Pedalboard**: High-quality audio effects
- **Demucs**: State-of-the-art source separation
- **AudioFlux**: Advanced audio transformations and analysis

## License

MIT
