'''
Basic example of using the audio_toolbox API for audio analysis and transformation.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_toolbox import AudioProcessor

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
    
    # Example 1: Analysis
    print("\nAnalyzing audio...")
    # Extract spectral features
    features = processor.analyze.spectral_features()
    print("Extracted spectral features")
    
    # Detect onsets
    onsets = processor.analyze.detect_onsets()
    print(f"Detected {len(onsets)} onsets")
    
    # Beat tracking
    beats, tempo = processor.analyze.track_beats()
    print(f"Detected tempo: {tempo:.1f} BPM with {len(beats)} beats")
    
    # Example 2: Apply effects
    print("\nApplying effects...")
    # Apply a chain of effects
    processor.effects.reverb()\
             .delay()\
             .gain()
    
    # Save processed audio
    processed_path = "output_effects.wav"
    processor.save(processed_path)
    print(f"Saved processed audio to {processed_path}")
    
    # Example 3: Source separation
    print("\nPerforming source separation...")
    # Load the original audio again
    processor = AudioProcessor(audio_file)
    
    # Separate vocals and accompaniment
    vocals, accompaniment = processor.separate.vocals()
    
    # Save both stems
    vocals_processor = AudioProcessor(audio_data=vocals, sample_rate=processor.sample_rate)
    vocals_processor.save("output_vocals.wav")
    
    accompaniment_processor = AudioProcessor(audio_data=accompaniment, sample_rate=processor.sample_rate)
    accompaniment_processor.save("output_accompaniment.wav")
    
    print("Saved separated stems to output_vocals.wav and output_accompaniment.wav")
    
    # Example 4: Transformations
    print("\nApplying transformations...")
    # Load the original audio again
    processor = AudioProcessor(audio_file)
    
    # Apply time stretching and pitch shifting
    processor.transform.time_and_pitch()\
              .normalize()\
              .fade_in(0.5)\
              .fade_out(1.0)
    
    # Save transformed audio
    processor.save("output_transformed.wav")
    print("Saved transformed audio to output_transformed.wav")
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()
