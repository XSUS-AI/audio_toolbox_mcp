'''
Example demonstrating audio source separation with different models and configurations.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_toolbox import AudioProcessor
from src.audio_toolbox.separate import SeparationRequest, TwoStemSeparationRequest

def create_plot(processor, title):
    """Create and save a plot of the audio waveform"""
    fig = processor.analyze.visualize_waveform()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
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
    
    # Create a plot of the original audio
    create_plot(processor, "Original Audio")
    
    # List available separation models
    available_models = processor.separate.available_models()
    print(f"\nAvailable separation models: {', '.join(available_models)}")
    
    # Default model
    default_model = "htdemucs"  # Change to a model that's in available_models if needed
    
    print(f"\nUsing model: {default_model}")
    
    # Example 1: Full separation into all stems
    print("\nSeparating audio into all stems...")
    request = SeparationRequest(model_name=default_model, device="cpu")
    stems = processor.separate.separate_stems(request)
    
    # Print the available stems
    print(f"Stems available: {', '.join(stems.keys())}")
    
    # Save each stem and create visualization
    for stem_name, stem_audio in stems.items():
        stem_processor = AudioProcessor(audio_data=stem_audio, sample_rate=processor.sample_rate)
        stem_processor.save(f"output_{stem_name}.wav")
        create_plot(stem_processor, f"{stem_name.capitalize()} Stem")
    
    print(f"Saved all stems to separate files")
    
    # Example 2: Vocal isolation with different configuration
    print("\nIsolating vocals with fine-tuned model...")
    
    # Use the fine-tuned model if available
    ft_model = "htdemucs_ft" if "htdemucs_ft" in available_models else default_model
    
    request = TwoStemSeparationRequest(
        model_name=ft_model,
        stem="vocals",
        shifts=2,  # Use 2 random shifts for better quality
        device="cpu"
    )
    
    vocals, accompaniment = processor.separate.vocals(request)
    
    # Save the isolated vocals and accompaniment
    vocals_processor = AudioProcessor(audio_data=vocals, sample_rate=processor.sample_rate)
    vocals_processor.save("output_vocals_high_quality.wav")
    create_plot(vocals_processor, "Vocals (High Quality)")
    
    accompaniment_processor = AudioProcessor(audio_data=accompaniment, sample_rate=processor.sample_rate)
    accompaniment_processor.save("output_accompaniment_high_quality.wav")
    create_plot(accompaniment_processor, "Accompaniment (High Quality)")
    
    print("Saved high-quality vocal separation")
    
    # Example 3: Apply processing chain to isolated stems
    print("\nApplying effects to isolated stems...")
    
    # Process the vocals with effects
    vocals_processor.effects.reverb().compressor().normalize()
    vocals_processor.save("output_vocals_processed.wav")
    create_plot(vocals_processor, "Processed Vocals")
    
    # Process the accompaniment with effects
    accompaniment_processor.effects.lowpass_filter().gain().normalize()
    accompaniment_processor.save("output_accompaniment_processed.wav")
    create_plot(accompaniment_processor, "Processed Accompaniment")
    
    print("Applied effects to isolated stems")
    
    # Example 4: Mix the processed stems back together
    print("\nMixing processed stems...")
    
    # Create a new audio array with the same shape as the original processed vocals
    mixed_audio = vocals_processor.audio_data + accompaniment_processor.audio_data
    
    # Avoid clipping by normalizing
    max_amp = np.max(np.abs(mixed_audio))
    if max_amp > 1.0:
        mixed_audio = mixed_audio / max_amp * 0.95  # Leave a little headroom
    
    # Create a new processor with the mixed audio
    mixed_processor = AudioProcessor(audio_data=mixed_audio, sample_rate=processor.sample_rate)
    mixed_processor.save("output_mixed_processed.wav")
    create_plot(mixed_processor, "Mixed Processed Audio")
    
    print("Saved mixed processed audio")
    
    # Example 5: Using a 6-stem model if available
    if "htdemucs_6s" in available_models:
        print("\nTrying 6-stem separation model...")
        request = SeparationRequest(model_name="htdemucs_6s", device="cpu")
        stems_6s = processor.separate.separate_stems(request)
        
        print(f"6-stem model stems: {', '.join(stems_6s.keys())}")
        
        # Process and save just the guitar if available
        if "guitar" in stems_6s:
            guitar_processor = AudioProcessor(audio_data=stems_6s["guitar"], sample_rate=processor.sample_rate)
            guitar_processor.save("output_guitar.wav")
            create_plot(guitar_processor, "Guitar Stem")
            print("Saved isolated guitar stem")
    
    print("\nAll separation examples completed successfully!")

if __name__ == "__main__":
    main()
