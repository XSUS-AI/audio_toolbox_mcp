'''
Example demonstrating audio transformations with the audio_toolbox.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_toolbox import AudioProcessor
from src.audio_toolbox.transform import (
    TimeStretchRequest, PitchShiftRequest, TimeAndPitchRequest, ResampleRequest
)

def create_visualization(processor, title):
    """Create and save a waveform and spectrogram visualization"""
    # Create waveform plot
    fig_wave = processor.analyze.visualize_waveform()
    fig_wave.suptitle(f"{title} - Waveform")
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_waveform.png")
    plt.close(fig_wave)
    
    # Create spectrogram plot
    fig_spec = processor.analyze.visualize_spectrogram(log_scale=True)
    fig_spec.suptitle(f"{title} - Spectrogram")
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_spectrogram.png")
    plt.close(fig_spec)

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
    
    # Create visualizations of the original audio
    create_visualization(processor, "Original")
    
    # Example 1: Time stretching
    print("\nApplying time stretching...")
    
    # Speed up the audio (0.5 = half duration, double speed)
    speedup_processor = AudioProcessor(audio_file)
    speedup_request = TimeStretchRequest(
        stretch_factor=2.0,  # Make twice as fast
        high_quality=True,
        preserve_formants=False,
        transient_mode="crisp"
    )
    speedup_processor.transform.time_stretch(speedup_request)
    speedup_processor.save("output_speedup.wav")
    create_visualization(speedup_processor, "Speed Up")
    print(f"Original duration: {processor.duration:.2f}s, Speed up duration: {speedup_processor.duration:.2f}s")
    
    # Slow down the audio (2.0 = double duration, half speed)
    slowdown_processor = AudioProcessor(audio_file)
    slowdown_request = TimeStretchRequest(
        stretch_factor=0.5,  # Make twice as slow
        high_quality=True,
        preserve_formants=False,
        transient_mode="smooth"
    )
    slowdown_processor.transform.time_stretch(slowdown_request)
    slowdown_processor.save("output_slowdown.wav")
    create_visualization(slowdown_processor, "Slow Down")
    print(f"Original duration: {processor.duration:.2f}s, Slow down duration: {slowdown_processor.duration:.2f}s")
    
    # Example 2: Pitch shifting
    print("\nApplying pitch shifting...")
    
    # Shift up (e.g., for chipmunk effect)
    pitch_up_processor = AudioProcessor(audio_file)
    pitch_up_request = PitchShiftRequest(
        semitones=4.0,  # Shift up by 4 semitones (major third)
        high_quality=True,
        preserve_formants=False
    )
    pitch_up_processor.transform.pitch_shift(pitch_up_request)
    pitch_up_processor.save("output_pitch_up.wav")
    create_visualization(pitch_up_processor, "Pitch Up")
    
    # Shift down (e.g., for deeper voice)
    pitch_down_processor = AudioProcessor(audio_file)
    pitch_down_request = PitchShiftRequest(
        semitones=-4.0,  # Shift down by 4 semitones (major third)
        high_quality=True,
        preserve_formants=False
    )
    pitch_down_processor.transform.pitch_shift(pitch_down_request)
    pitch_down_processor.save("output_pitch_down.wav")
    create_visualization(pitch_down_processor, "Pitch Down")
    
    print("Saved pitch-shifted outputs")
    
    # Example 3: Combined time and pitch transformations
    print("\nApplying combined time and pitch transformations...")
    
    # Change both time and pitch
    combined_processor = AudioProcessor(audio_file)
    combined_request = TimeAndPitchRequest(
        stretch_factor=1.2,       # Slightly slower
        pitch_shift_semitones=2,  # Slightly higher pitch
        high_quality=True,
        preserve_formants=True    # Try to maintain voice characteristics
    )
    combined_processor.transform.time_and_pitch(combined_request)
    combined_processor.save("output_combined.wav")
    create_visualization(combined_processor, "Combined")
    
    print("Saved combined time and pitch transformation")
    
    # Example 4: Resampling
    print("\nApplying resampling...")
    
    # Resample to lower quality
    lofi_processor = AudioProcessor(audio_file)
    lofi_request = ResampleRequest(
        target_sample_rate=8000,  # 8 kHz like old telephone quality
        quality="low"
    )
    lofi_processor.transform.resample(lofi_request)
    lofi_processor.save("output_lofi.wav")
    create_visualization(lofi_processor, "Lo-Fi Resampling")
    
    # Resample to higher quality if original is less than 48kHz
    if processor.sample_rate < 48000:
        hifi_processor = AudioProcessor(audio_file)
        hifi_request = ResampleRequest(
            target_sample_rate=48000,  # 48 kHz high quality
            quality="high"
        )
        hifi_processor.transform.resample(hifi_request)
        hifi_processor.save("output_hifi.wav")
        create_visualization(hifi_processor, "Hi-Fi Resampling")
    
    print("Saved resampled outputs")
    
    # Example 5: Other transformations
    print("\nApplying other transformations...")
    
    # Normalize audio
    normalize_processor = AudioProcessor(audio_file)
    normalize_processor.transform.normalize(target_db=-1.0)
    normalize_processor.save("output_normalized.wav")
    create_visualization(normalize_processor, "Normalized")
    
    # Reverse audio
    reverse_processor = AudioProcessor(audio_file)
    reverse_processor.transform.reverse()
    reverse_processor.save("output_reversed.wav")
    create_visualization(reverse_processor, "Reversed")
    
    # Fade in and out
    fade_processor = AudioProcessor(audio_file)
    fade_processor.transform.fade_in(duration_seconds=1.0).fade_out(duration_seconds=2.0)
    fade_processor.save("output_faded.wav")
    create_visualization(fade_processor, "Faded")
    
    # Trim silence
    trim_processor = AudioProcessor(audio_file)
    trim_processor.transform.trim_silence(threshold_db=-40.0, pad_seconds=0.2)
    trim_processor.save("output_trimmed.wav")
    create_visualization(trim_processor, "Trimmed")
    
    print("Saved additional transformation outputs")
    
    # Example 6: Chain multiple transformations
    print("\nApplying chained transformations...")
    
    chain_processor = AudioProcessor(audio_file)
    # Apply a series of transformations using method chaining
    chain_processor.transform.pitch_shift(PitchShiftRequest(semitones=2.0))\
                    .time_stretch(TimeStretchRequest(stretch_factor=0.8))\
                    .normalize()\
                    .fade_in(0.5)\
                    .fade_out(0.5)
    
    chain_processor.save("output_transform_chain.wav")
    create_visualization(chain_processor, "Transform Chain")
    
    print("Saved chained transformation output")
    
    print("\nAll transformation examples completed successfully!")

if __name__ == "__main__":
    main()
