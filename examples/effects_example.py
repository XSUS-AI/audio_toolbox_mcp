'''
Example demonstrating audio effects processing with the audio_toolbox.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_toolbox import AudioProcessor
from src.audio_toolbox.effects import (
    ReverbRequest, DelayRequest, ChorusRequest, PhaserRequest,
    CompressorRequest, DistortionRequest, GainRequest, PitchShiftRequest,
    FilterRequest, NoiseGateRequest, LimiterRequest, ChainRequest
)

def create_spectrogram(processor, title):
    """Create and save a spectrogram plot"""
    fig = processor.analyze.visualize_spectrogram(log_scale=True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_spectrogram.png")
    plt.close(fig)

def create_waveform(processor, title):
    """Create and save a waveform plot"""
    fig = processor.analyze.visualize_waveform()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_waveform.png")
    plt.close(fig)

def main():
    # Path to an audio file (replace with your own file)
    audio_file = "input.mp3"
    
    if not os.path.exists(audio_file):
        print(f"Please place an audio file at {audio_file} or update the path in the script.")
        return
    
    # Create an AudioProcessor instance
    processor = AudioProcessor(audio_file)
    print(f"Loaded audio: {audio_file}")
    print(f"Duration: {processor.duration:.2f} seconds")
    print(f"Sample rate: {processor.sample_rate} Hz")
    print(f"Channels: {processor.num_channels}")
    
    # Save original visualization for comparison
    create_waveform(processor, "Original")
    create_spectrogram(processor, "Original")
    
    # Example 1: Reverb effect with custom parameters
    print("\nApplying reverb effect...")
    reverb_processor = AudioProcessor(audio_file)
    reverb_request = ReverbRequest(
        room_size=0.8,
        damping=0.5,
        wet_level=0.6,
        dry_level=0.2,
        width=1.0,
        freeze_mode=0.0
    )
    reverb_processor.effects.reverb(reverb_request)
    reverb_processor.save("output_reverb.wav")
    
    create_spectrogram(reverb_processor, "Reverb")
    print("Saved reverb effect output")
    
    # Example 2: Delay effect with custom parameters
    print("\nApplying delay effect...")
    delay_processor = AudioProcessor(audio_file)
    delay_request = DelayRequest(
        delay_seconds=0.3,
        feedback=0.4,
        mix=0.5
    )
    delay_processor.effects.delay(delay_request)
    delay_processor.save("output_delay.wav")
    
    create_waveform(delay_processor, "Delay")
    print("Saved delay effect output")
    
    # Example 3: Chorus effect
    print("\nApplying chorus effect...")
    chorus_processor = AudioProcessor(audio_file)
    chorus_request = ChorusRequest(
        rate_hz=1.5,
        depth=0.7,
        centre_delay_ms=7.0,
        feedback=0.4,
        mix=0.5
    )
    chorus_processor.effects.chorus(chorus_request)
    chorus_processor.save("output_chorus.wav")
    
    create_spectrogram(chorus_processor, "Chorus")
    print("Saved chorus effect output")
    
    # Example 4: Distortion effect
    print("\nApplying distortion effect...")
    distortion_processor = AudioProcessor(audio_file)
    distortion_request = DistortionRequest(drive_db=15.0)
    distortion_processor.effects.distortion(distortion_request)
    distortion_processor.save("output_distortion.wav")
    
    create_waveform(distortion_processor, "Distortion")
    create_spectrogram(distortion_processor, "Distortion")
    print("Saved distortion effect output")
    
    # Example 5: Filter effects
    print("\nApplying filter effects...")
    
    # Low-pass filter
    lowpass_processor = AudioProcessor(audio_file)
    lowpass_request = FilterRequest(cutoff_frequency_hz=2000.0)
    lowpass_processor.effects.lowpass_filter(lowpass_request)
    lowpass_processor.save("output_lowpass.wav")
    
    create_spectrogram(lowpass_processor, "Low-Pass Filter")
    
    # High-pass filter
    highpass_processor = AudioProcessor(audio_file)
    highpass_request = FilterRequest(cutoff_frequency_hz=500.0)
    highpass_processor.effects.highpass_filter(highpass_request)
    highpass_processor.save("output_highpass.wav")
    
    create_spectrogram(highpass_processor, "High-Pass Filter")
    
    print("Saved filter effect outputs")
    
    # Example 6: Effect chains
    print("\nApplying effect chains...")
    
    # Create a chain request with multiple effects
    chain_request = ChainRequest(
        effects=[
            {"type": "compressor", "threshold_db": -20.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 100.0},
            {"type": "reverb", "room_size": 0.6, "damping": 0.4, "wet_level": 0.3, "dry_level": 0.7},
            {"type": "delay", "delay_seconds": 0.2, "feedback": 0.3, "mix": 0.25},
            {"type": "gain", "gain_db": 3.0}
        ]
    )
    
    chain_processor = AudioProcessor(audio_file)
    chain_processor.effects.chain(chain_request)
    chain_processor.save("output_chain.wav")
    
    create_waveform(chain_processor, "Effect Chain")
    create_spectrogram(chain_processor, "Effect Chain")
    
    print("Saved effect chain output")
    
    # Example 7: Custom method chaining
    print("\nApplying custom effect chain with method chaining...")
    
    custom_chain_processor = AudioProcessor(audio_file)
    # Apply a series of effects with custom parameters
    custom_chain_processor.effects.compressor(CompressorRequest(threshold_db=-25.0, ratio=3.0))\
                          .delay(DelayRequest(delay_seconds=0.15, feedback=0.2, mix=0.3))\
                          .chorus(ChorusRequest(rate_hz=2.0, depth=0.5))\
                          .gain(GainRequest(gain_db=2.0))
    
    custom_chain_processor.save("output_custom_chain.wav")
    
    create_waveform(custom_chain_processor, "Custom Chain")
    create_spectrogram(custom_chain_processor, "Custom Chain")
    
    print("Saved custom chain output")
    
    print("\nAll effect examples completed successfully!")

if __name__ == "__main__":
    main()
