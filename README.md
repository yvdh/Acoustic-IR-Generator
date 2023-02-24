# Acoustic instrument IR Generator
A script for Scilab that allows generation of an IR from pickup and mic'd audio of an acoustic instrument. It does NOT generate it's own excitation signals (sine, chirps,  pink noise, ...) but uses a musical performance as 'test' signal.

This runs on Scilab (an open Matlab-like computing environment), and has been tested with version 6.1.1.

## What does it do?

The script will try to generate an impulse response that transforms the source audio into the target audio by time-domain convolution (or frequency domain multiplication). Some of the features included in the algorithm are:
- minimum phase transform: ensures low latency at the cost of some phase shifts.
- Wiener correction: uses signal-to-noise ratio to avoid huge boosts in parts of the spectrum where there is basically no signal but only noise.
- transfer function smoothing in per-octave bands.
- normalization of the IR when saving to [0.0 - 1.0]. Float WAV files can save values > 1, but most IR loaders will either clip or normalize them when loading. Doing this here allows an estimation of the correction the IR loader will have to perform. It's sad that the huge dynamic range of the float WAV format cannot be used ....

## How to use it?

### Environment 

1. Install Scilab if needed. I used 6.1.1.
2. Download the two files in the repository into a folder.
3. Set the working directory of scilab to that folder.
4. Run the UI part of the script in the scilab command processor using 'exec AudioIR.ui.sce'.
5. Select the source and target audio files (top button left)
6. Adjust the settings if needed.
7. Compute the IR.
8. Audition the result using the 'Play' buttons. The transformed audio should sound very similar to the target audio ...
9. Save the IR, or adjust settings and recompute.

### Audio files

The script requires two audio files in WAV format, of the same sampling rate (e.g. 48 kHz) and length.
The 'source' represents the signal from the pickup of your acoustic instrument, while the 'target' is a properly mic'd signal of that instrument. Both represent the SAME performance, so they should have been recorded together. The performance should:
- start with 10 s silence (for noise estimation).
- last 60 - 90 s.
- contain a good representation of the instrument and your playing style (strumming, fingerpicking, ...).

I am a physicist and biomedical engineer with some knownledge of signal processing, but by no means an expert! 

'My' algorithm is just a pulling together established and known methods, and I make no claim to originality or innovation other than some minor tweaks here and there. I have tried to acknowledge other authors were possible in the script comments. Most notably the idea of using Scilab is similar to that used by 'Cuki' in his IR generator (I used Matlab a lot, but that is a closed system).

All constructive comments and suggestions welcome!

![Main](https://user-images.githubusercontent.com/16563417/221225835-6f69f463-ced9-45d8-9778-9aa7117bbc2b.jpg)
![PSDs](https://user-images.githubusercontent.com/16563417/221225854-4d422959-84bf-4c87-91a7-a45e2359aeb5.jpg)
![IRAndTF](https://user-images.githubusercontent.com/16563417/221225859-78e7bcd1-355d-403b-b3d4-7d07fe9d79d9.jpg)


