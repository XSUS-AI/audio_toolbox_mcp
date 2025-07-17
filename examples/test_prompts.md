# Example Prompts for AudioProcessor Agent

This document contains example prompts to use with the AudioProcessor agent, demonstrating various capabilities from simple to complex operations.

## Basic Operations

### Loading and Analyzing Audio

```
Load my song.mp3 and tell me its basic characteristics.
```

```
Load vocal_take.wav and analyze it for pitch accuracy and timing.
```

### Simple Effects

```
Load guitar.wav and add a reverb effect.
```

```
Load drums.wav and make them more punchy with compression.
```

### Basic Transformations

```
Load voice.wav and pitch shift it up by 2 semitones.
```

```
Load track.mp3 and slow it down by 5% without changing the pitch.
```

## Intermediate Operations

### Audio Separation

```
Load my song.mp3 and extract just the vocals.
```

```
Load band_recording.wav, separate the drums, and save them as a separate file.
```

### Effect Chains

```
Load bass.wav, add compression with a 4:1 ratio, then add some distortion and EQ to boost the low-mids.
```

```
Load acoustic_guitar.wav, add some gentle compression, then a warm reverb, and finally a slight stereo widening.
```

### Multiple Transformations

```
Load drums.wav, increase the tempo by 10%, then add some compression to make them louder, and finally apply a room reverb.
```

```
Load synth.wav, apply a flanger effect, then some delay with moderate feedback, and finally normalize the output.
```

## Advanced Operations

### Complex Audio Processing Chains

```
Load vocals.wav and apply a professional vocal chain: first a high-pass filter at 100Hz, then a compressor with -20dB threshold and 3:1 ratio, followed by a de-esser, and finally a plate reverb with 30% wet signal.
```

```
Load guitar_solo.wav and create a spacey effect by first applying a pitch shifter to create harmonies a fifth above, then adding ping-pong delay with 300ms delay time, then a phaser effect with slow rate, and finally a large hall reverb with long decay time.
```

### Multi-step Separation and Processing

```
Load song.mp3, separate the vocals, drums, bass, and other instruments. Apply different effects to each: add reverb to the vocals, compression to the drums, distortion to the bass, and a chorus effect to the other instruments. Then mix them back together with the vocals slightly louder than the rest.
```

```
Load band_recording.wav, identify the tempo and key, then separate it into stems. Quantize the drums to fix timing issues, apply pitch correction to the vocals, add compression to the bass, and some reverb to the guitars. Then mix everything back together and normalize the result.
```

### Creative Sound Design

```
Load piano.wav and transform it into an atmospheric pad sound by stretching it to 3x the original length, applying a chorus effect with high depth, adding a slow phaser, and finally a large reverb with 5 second decay. Save the result as ambient_pad.wav.
```

```
Load drum_loop.wav and create a glitchy version by first chopping it into 16 equal segments, randomly rearranging 50% of them, applying a bitcrusher effect with 8-bit depth, adding some digital distortion, and finally a stereo delay with different timing on left and right channels.
```

## Professional Workflows

### Complete Mixing Session

```
I have a full mix project. Load drums.wav, bass.wav, guitars.wav, and vocals.wav. For the drums, apply compression with a 4:1 ratio and boost around 5kHz for snap. For the bass, add a touch of distortion and boost around 80Hz while cutting at 300Hz. For the guitars, add a moderate room reverb and pan one take 30% left and the other 30% right. For the vocals, apply compression, de-essing, a slight EQ boost around 3kHz for presence, and a plate reverb. Mix all tracks together with drums at 90% volume, bass at 100%, guitars at 85%, and vocals at 95%. Apply a gentle mastering chain with multiband compression and bring the final output to -1dB peak level.
```

### DJ-style Mashup

```
Load song1.mp3 and song2.mp3. Analyze their tempos and keys. Time-stretch and pitch-shift song2 to match song1. Take the intro from song1, create a transition into the drop from song2, then go back to the breakdown of song1, and finish with the outro of song2. Add a filter sweep during each transition and some reverb on the final note of each section. Save the result as mashup.wav.
```

### Sound Design for Film

```
I'm creating sound for a sci-fi scene. Load spaceship_ambience.wav and enhance the low end rumble by boosting frequencies below 100Hz. Then add some movement with a slow phaser. Separately, load mechanical_clicks.wav, apply a pitch shift down 5 semitones, add some metallic reverb, and randomly pan the sounds across the stereo field. Finally, load whoosh.wav, stretch it to 5 seconds, apply a rising filter sweep, and add some distortion. Layer all three processed sounds together with the ambience as the base, the clicks at 60% volume, and the whoosh fading in from 0% to 100% over its duration.
```

## Specialized Tasks

### Podcast Production

```
Load podcast_interview.wav, identify and remove sections of silence longer than 2 seconds, apply a noise reduction to clean up background noise, add compression to even out voice levels, and finally add a subtle EQ curve to enhance vocal clarity.
```

### Audio Restoration

```
Load old_vinyl_recording.wav and restore it by first applying a de-click algorithm to remove pops, then a de-noise process to reduce hiss, followed by gentle compression to restore dynamics, and finally an EQ to enhance the muffled high end without increasing noise.
```

### Spatial Audio Processing

```
Load orchestra.wav and create a binaural version that simulates the positions of instruments in a concert hall: place the strings on the left, brass on the right, woodwinds in the center-left, and percussion in the center-right, all with appropriate distances from the listener. Add a concert hall reverb that matches this spatial positioning.
```

## Analysis and Machine Learning

```
Load song_collection/*.mp3 and analyze all files to identify common tempo ranges, key signatures, and dynamic patterns. Create a visualization showing the distribution of these characteristics across the collection.
```

```
Load reference_track.wav and my_mix.wav. Compare them in terms of spectral balance, stereo width, dynamics, and loudness. Suggest processing steps to make my_mix.wav more similar to the reference in terms of these characteristics.
```

## Feedback and Troubleshooting

If you encounter any issues or have suggestions for improving these example prompts, please let us know. These examples are designed to showcase the capabilities of the AudioProcessor agent and inspire creative applications.
