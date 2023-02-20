// Audio IR computation from a source and target file (usually a pickup and a mic'd signal).
// Yves Vander Haeghen, december 2021, december 2022.
// Version 1.0 (3/1/2023)
// MIT license.

// Compute the IR given a source and target audio file. If seperate source and target noise files are passed on they will be used, else the noise will be estimated from 'silent'
// parts of the source and target audio files.
// The impulseResponse can be postprocessed with either a simply trunctation or a minimum-phase transform with truncation.
// The linear phase IR (i.e. no postprocesing) )is the most correct solution, 
// but it is non-causal and cannot be implemented without introducing latency by shifting it's time response to make it causal.
// Truncating this by removing the part before the main delayed peak removes information and makes it the frequency match less correct, especially in the low frequencies.
// The minimum phase transform has almost zero latency but introduces some phase shifts in the solution. It does however retain a good match in the low frequencies.
// The computedTargetAudio contains the IR applied to the source audio.
function [impulseResponse, sourceAudio, targetAudio, computedTargetAudio, sampleRate, IRGain] = AudioIR(options, sourceAudioFile, targetAudioFile, sourceNoiseAudioFile, targetNoiseAudioFile)
    // Display options
    mprintf("Computing Impulse response from source and target audio, by Yves Vander Haeghen. V1.0 (28/11/2021), V1.1 (4/12/2022)\n");
    mprintf("Options\n")
    disp(options)
    mprintf("Input files\n")
    disp(sourceAudioFile)
    disp(targetAudioFile)
    
    [leftArgs, rightArgs] = argn(0);
    
    if (rightArgs < 2) then
        // For testing: create synthetic signals
        sampleRate = 44100;
        // Signals are at -6 dB (200 Hz), -12 dB (500 Hz) and -9 dB (2000Hz) in the source. Noise is at -42 dB
        // These should generate a transfer function with a -6 dB at 200 Hz, +6 dB at 500 Hz and -12 dB at 2000 Hz.
        sourceAudio = GenerateSine(4, 200, 0.5, sampleRate) + GenerateSine(4, 500, 0.25, sampleRate) + GenerateSine(4, 2000, 0.35, sampleRate);
        targetAudio = GenerateSine(4, 200, 0.25, sampleRate) + GenerateSine(4, 500, 0.5, sampleRate) + GenerateSine(4, 2000, 0.0625, sampleRate);
        sourceNoiseAudio = GenerateNoise(4, 0.0078125, sampleRate); 
        targetNoiseAudio = GenerateNoise(4, 0.0078125, sampleRate); 
        
        // Make sure the noise is present in the signals!
        sourceAudio = sourceAudio + sourceNoiseAudio;
        targetAudio = targetAudio + targetNoiseAudio;        
    else
        // Load the audio files. We assume same sample rate and length.
        // Also make sure we are dealing with mono, transforming them if necessary.
        [sourceAudio, sourceAudioSampleRate] = wavread(sourceAudioFile);
        [targetAudio, targetAudioSampleRate] = wavread(targetAudioFile);
        
        // Select channel (left or right). If only one channel present that will be used.
        //sourceAudio = sourceAudio(1,:); // To mono LEFT
        //targetAudio = targetAudio(1,:); // To mono LEFT
        sourceAudio = GetMonoAudio(sourceAudio, options.channel); 
        targetAudio = GetMonoAudio(targetAudio, options.channel); 
        
        // Some checking on the files ...
        if (length(sourceAudio) ~= length(targetAudio)) then 
           error(mprintf('Length of the source and target audio files are not the same: %d vs %d', length(sourceAudio), length(targetAudio)));
           abort 
        end
        if (sourceAudioSampleRate ~= targetAudioSampleRate) then 
           error(mprintf('Sample rates of the source and target audio files are the same: %d , %d , %d, %d', sourceAudioSampleRate, targetAudioSampleRate));
           abort 
        end
        sampleRate = sourceAudioSampleRate;  
    
        // Determine noise in audio.    
        if (rightArgs == 5) then
            // Load noise files and make sure we have mono.
            [sourceNoiseAudio, sourceAudioNoiseSampleRate] = wavread(sourceNoiseAudioFile);
            [targetNoiseAudio, targetAudioNoiseSampleRate] = wavread(targetNoiseAudioFile);
            sourceNoiseAudio = sourceNoiseAudio(1,:); // To mono
            targetNoiseAudio = targetNoiseAudio(1,:); // To mono
            sourceNoiseAudioSampleRate = sampleRate;
            targetNoiseAudioSampleRate = sampleRate;
                       
            if ((sourceNoiseAudioSampleRate ~= targetNoiseAudioSampleRate) | (sourceNoiseAudioSampleRate ~= sampleRate))then 
               error(mprintf('Sample rates of the audio files are not equal : %d , %d , %d, %d', sourceAudioSampleRate, targetAudioSampleRate, sourceAudioNoiseSampleRate,targetAudioNoiseSampleRate));
               abort 
            end
        else
            // Compute noise if not provided in seperate files.
            // Create noise files from silent parts in the source and target files.
            mprintf('Estimating noise using the source and target audio\n' )
            [sourceNoiseAudio, sourceAudio, splitIndex] = SplitNoiseAndSignal(sourceAudio, options.signalNoiseSplitTimeWindow, options.signalNoiseSplitDBFS, sourceAudioSampleRate);
            
            // Split the second file in the same way!   
            targetNoiseAudio = targetAudio(1:splitIndex);
            targetAudio = targetAudio(splitIndex+1:$);            
        end
    
    end
    mprintf("Audio sample rate: %d Hz\n", sampleRate);
    
    // Compute the minimum length of an IR using the mimimum desired IR length in ms.
    // This will be used when truncating the IR's.
    // Note that increasing this will also allow inclusion of 'late' information like short room reverberation. 
    // E.g. a value of 200 ms at 48 kHz sampling (9600 pt IR) should allow ~ 10 Hz signals to be represented, and possibly 
    // some short room reverb (gross simplification) and instrument resonances.
    // Aim for say 80 ms minimum or acoustic instruments?
    // Note that some IR loaders do not allow such long IR's ...
    minIRLength = sampleRate * options.impulseResponseTimeLengthMS / 1000;
     
    mprintf("Impulse response length should be at least %d points to be %d ms long \n", minIRLength, options.impulseResponseTimeLengthMS);
     
    // Compute RMS of the input signals without the start noise. Abort if too low ...
    sourceAudioAvgRMS = sqrt(mean(sourceAudio .^2));
    targetAudioAvgRMS = sqrt(mean(targetAudio .^2));
    sourceAudioDBFS = AmplitudeToDB(sourceAudioAvgRMS, 0.707);
    targetAudioDBFS = AmplitudeToDB(targetAudioAvgRMS, 0.707);
    mprintf("Source audio RMS: %f (%f dBFS)\n", sourceAudioAvgRMS, sourceAudioDBFS);
    mprintf("Target audio RMS: %f (%f dBFS)\n", targetAudioAvgRMS, targetAudioDBFS);
    if (options.minimumAudioDBFS > sourceAudioDBFS) then
        error('Source signal level too low!');
        abort;
    end
    if (options.minimumAudioDBFS > targetAudioDBFS) then
        error('Target signal levels too low!');
        abort;
    end
    
    // Create time vectors
    time = [1:length(sourceAudio)]./sampleRate;
    noiseTime = [1:length(sourceNoiseAudio)]./sampleRate;
    
    // Plot audio. Noise is usually very short ...
    plotTimeIndex = [1:max(min(options.audioPlotTime*sampleRate, length(time)), length(noiseTime))];   
    if (options.plotAudio == 'on') then
        audioPlotHandle = figure();
        subplot(1,2,1)
        plot(time(plotTimeIndex), sourceAudio(plotTimeIndex), 'b');
        title('Source audio  and noise')
        xlabel('Time (s)')
        ylabel('Amplitude')
        set(gca(),"grid",[1 1])
        set(gca(),"auto_clear","off") 
        plot(noiseTime, sourceNoiseAudio, 'c');    
        xlabel('Time (s)')
        ylabel('Amplitude')
        subplot(1,2,2)
        plot(time(plotTimeIndex),targetAudio(plotTimeIndex),'r'); 
        title('Target audio and noise ')
        xlabel('Time (s)')
        ylabel('Amplitude')
        set(gca(),"grid",[1 1])
        set(gca(),"auto_clear","off") 
        plot(noiseTime,targetNoiseAudio,'y'); 
        xlabel('Time (s)')
        ylabel('Amplitude')        
    end
    
    // Compute Power spectra (Welch with Hanning window). Result are two-sided for computations, one sided for plots.
    sourceSpectrum = PowerSpectrum(sourceAudio, options.fftWindowSize, options.fftHopSize, 'hn', sampleRate);
    targetSpectrum = PowerSpectrum(targetAudio, options.fftWindowSize, options.fftHopSize, 'hn', sampleRate);
    sourceNoiseSpectrum = PowerSpectrum(sourceNoiseAudio, options.fftWindowSize, options.fftHopSize, 'hn',sampleRate);
    targetNoiseSpectrum = PowerSpectrum(targetNoiseAudio, options.fftWindowSize, options.fftHopSize, 'hn',sampleRate);
                   
    // Plot spectra. We need to leave out the DC component for log plots ...
    // We hide anything below ~ 70 Hz ...
    frequency = FoldedFFTFrequencies(options.fftWindowSize, sampleRate);
    plotSpectrumIndex = find(frequency > options.plotCutoffFrequency); // Always leave out DC!    
    if (options.plotSpectra == 'on') then
        audioSpectraPlotHandle = figure();
        set(gca(),"grid",[1 1])
        set(gca(),"auto_clear","off") 
        semilogx(frequency(plotSpectrumIndex), PowerToDB(FoldFFTFrequencyDomainData(sourceSpectrum)(plotSpectrumIndex)),'b');
        title('Power spectral densities ')
        xlabel('Frequency (Hz)')
        ylabel('dbFS')
        semilogx(frequency(plotSpectrumIndex), PowerToDB(FoldFFTFrequencyDomainData(sourceNoiseSpectrum)(plotSpectrumIndex)),'b:');
        semilogx(frequency(plotSpectrumIndex), PowerToDB(FoldFFTFrequencyDomainData(targetSpectrum)(plotSpectrumIndex)), 'r'); 
        semilogx(frequency(plotSpectrumIndex), PowerToDB(FoldFFTFrequencyDomainData(targetNoiseSpectrum)(plotSpectrumIndex)), 'r:'); 
    end
       
    // Compute transfer function.
    // x is typically the instrument pickup, while y is the desired acoustic recording using a microphone.
    // First we have the naieve deconvolution, and it's VERY sensitive to SNR of x (* is convolution)
    // y(t) = IR(t) * x(t) =>  T(f) = y(f) / x(f), with x(f) and y(f) the Fourier spectra of x(t) and y(t).
    // Call this H0
    // Then there are other estimators for the transfer function, based on cross correlation between the target y and source x:
    // T(f) = Sxy(f) / Sxx(f) (noise not correlated with x): H1
    // T(f) = Syy(f) / Sxy(f) (noise not correlated with y): H2
    // In our case the noise is mainly on the target spectrum y, so we should use H0 or H1 => Sxy(f) / Sxx(f). H0 works best!
    // It's two sided, with DC at index 0, and Nyquist at N/2+1.
    if (options.transferFunctionEstimator == 'H0') then
        // Remember we obtained power spectra and we want something that works on amplitudes!
        transferFunction = sqrt(targetSpectrum ./ sourceSpectrum);
    elseif  (options.transferFunctionEstimator == 'H1') then
        crossSpectrum = CrossPowerSpectrum(sourceAudio, targetAudio, options.fftWindowSize, options.fftHopSize, 'hn', sampleRate);
        transferFunction = sqrt(crossSpectrum ./ sourceSpectrum);
    elseif   (options.transferFunctionEstimator == 'H2') then
        crossSpectrum = CrossPowerSpectrum(sourceAudio, targetAudio, options.fftWindowSize, options.fftHopSize, 'hn', sampleRate);
        transferFunction = sqrt(targetSpectrum ./ crossSpectrum);
    end
    
    // The Wiener formulation takes SNR into account to limit contributions from frequencies with low SNR of x.
    // SNR = xn(f) / x(f)
    // T'(f) = T(f) . (1 / (1 + 1 / (T(f)^2 . SNR)))
    if (options.WienerCorrection == 'on') then             
        SNR = sourceSpectrum ./ sourceNoiseSpectrum;
        weighting = (1 ./ (1 + 1 ./(transferFunction .* transferFunction .* SNR)));
        transferFunction = transferFunction .* weighting;
    end
    
    // Smooth the transferfunction using Octave bands.
    // This ensures a much better mininum phase transformation!
    // We need to order the fft according to ascending frequencies ...
    if (options.transferFunctionSmoothing > 0) then
        transferFunction = ifftshift(SmoothByOctaveBand(ShiftedFFTFrequencies(options.fftWindowSize, sampleRate), fftshift(transferFunction), options.transferFunctionSmoothing));
    end

    // Compute a minimum phase version of the transfer functionmif required.
    if (options.transferFunctionMinimumPhaseTransform == 'on') then
        mprintf("Computing min-phase transfer function ...\n");
        transferFunction = MinPhaseTransferFunction(transferFunction, -100);        
    end
    
    // Compute IR using inverse fourier transform. Basically this is a non-causal even (symmetric around t = 0,linear-phase) filter of rather long length fftWindowSize.
    // Shifting it to the right (just by convention) so that time becomes positive everywhere makes it causal, but linear phase (symmetric) and with a 
    // large delay of half the IR length = 0.5 . fftWindowSize / sampleRate.
    // This can be used in offline use cases (shift resulting signal back in time by the amount of delay), but not in any real-time use case.
    // It's also usually too long for real-time use.
    mprintf("Computing impulse response from transfer function ...\n");
    impulseResponse = ifftshift(ifft(transferFunction));        

    // Truncate the impulse response if required.
    // This minimizes the delay, but can remove a lot of information if a minimum phase transform was not applied before this.
    // It also removes unnecessary tails with almost no information in them. This can affect the lowest frequencies ...
    if (options.truncateImpulseResponse == 'on') then
        mprintf("Truncating impulse response to minimize delay and length ...\n");
        impulseResponse = TruncateImpulseResponse(ifftshift(ifft(transferFunction)), ...
                                                            options.impulseResponseTruncateThresholdDBFS, ...
                                                            options.impulseResponseTruncateThresholdHysteresisDBFS, ...
                                                            options.impulseResponseTruncateWindowSize, ...
                                                            minIRLength);
    end
     
    // Compute audio from source using the impulse response. These are not of the same length!
     computedTargetAudio = convol(impulseResponse, sourceAudio);

    // Compute the avegage RMS of the computed audio.
    computedTargetAudioRMS = sqrt(mean(computedTargetAudio .^2));
    computedTargetAudioDBFS = AmplitudeToDB(computedTargetAudioRMS, 0.707);    
    mprintf("Target audio * IR RMS: %f (%f dBFS)\n", computedTargetAudioRMS, computedTargetAudioDBFS);

     // Plot the transfer function and the impulse responses in a seperate window.
     if (options.plotTransferFunctionAndImpulseResponses == 'on') then
         transerFunctionImpulseResponsePlotHandle = figure();
         subplot(2,1,1)
         set(gca(),"auto_clear","off") 
         semilogx(frequency(plotSpectrumIndex), AmplitudeToDB(transferFunction(plotSpectrumIndex)), 'g');     
         title('Transfer function')
         xlabel('Frequency (Hz)')
         ylabel('dbFS')
         set(gca(),"grid",[1 1])
         set(gca(),"auto_clear","off") 
         subplot(2,1,2)
         set(gca(),"grid",[1 1])
         set(gca(),"auto_clear","off") 
         plot(time(1:length(impulseResponse))* 1000, impulseResponse, 'g'); 
         title('Impulse response')
         xlabel('Time (ms)')
         ylabel('Amplitude')
    end
     
     // Plot the computed PSD's in the spectra figure window
     if (options.plotSpectra == 'on') then
         scf(audioSpectraPlotHandle)
         semilogx(frequency(plotSpectrumIndex), PowerToDB(FoldFFTFrequencyDomainData(PowerSpectrum(computedTargetAudio, ... 
                options.fftWindowSize, options.fftHopSize, 'hn', sampleRate))(plotSpectrumIndex)),'g');
         legend(['Source';'Source noise';'Target';'Target noise';'Source * IR';]);
    end

     // Normalize impulse response. This is required because Scilab (and most IR loaders) cannot handle values outside [-1.0, 1.0].
     // This is the main reason why it will be necessary to apply large amounts of gain in the IR loader (which hopefully does floating point math!)
     maximumAmplitude = 0.99;
     maxIRValue = max(abs(impulseResponse));         
     IRGain = AmplitudeToDB(maxIRValue);
     
     if (options.normalizeImpulseResponse == 'on') then         
         mprintf("Add a gain of %f (%f dBFS) in the IR loader for the IR to obtain the same volume as the source signal\n", maxIRValue, IRGain);
         impulseResponse = maximumAmplitude * impulseResponse ./ maxIRValue;         
     else
         mprintf("A gain of %f (%f dBFS) maybe needed if the IR loader normalizes to obtain the same volume as the source audio \n", maxIRValue, IRGain);
     end

    // Apply gain to source audio to bring it up to the target level for accurate comparisons with computed audio.
    sourceGain = targetAudioAvgRMS / sourceAudioAvgRMS;
    mprintf("Added a gain of %f (%f dBFS) in the source audio to obtain the same volume for more accurate comparison to computed audio and target\n", sourceGain, AmplitudeToDB(sourceGain));
    sourceAudio = sourceAudio * sourceGain;    
endfunction

// Split an audio signal into a noise and signal part.
// We assume the signal part is at the start.
function [noise, signal, splitIndex] = SplitNoiseAndSignal(audio, splitWindowLength, splitDBFS, sampleRate)
    mprintf("Splitting audio into noise and signal using %d s windows and a %d dbFS threshold \n", splitWindowLength, splitDBFS );
       
    //Compute RMS for contiguous windows over whole audio
   nrSamplesPerWindow = splitWindowLength * sampleRate;
   nrWindows = floor(length(audio) / nrSamplesPerWindow);
   audioRMS = zeros(nrWindows);
   for windowNr = 1:nrWindows
       audioRMS(windowNr) = AmplitudeToRMS(audio, (windowNr - 0.5) * nrSamplesPerWindow + 1, nrSamplesPerWindow);
       // audioRMS(windowNr) = sqrt(mean(audio([((windowNr - 1) * nrSamplesPerWindow + 1):(windowNr * nrSamplesPerWindow + 1)]).^2));
   end
   
   // Find maximum, and compute threshold for noise.
   maxAudioRMS = max(audioRMS);
   noiseThresholdRMS = DBToAmplitude(splitDBFS, maxAudioRMS);

    // DEBUG
    mprintf("Maximum window audio RMS: %f, noise must be < %f\n", maxAudioRMS, noiseThresholdRMS);
   
   // Select all windows at the start with values > noise threshold.
   firstSignalWindow = find(audioRMS > noiseThresholdRMS)(1)-1;
   splitIndex = firstSignalWindow * nrSamplesPerWindow;
   
    // Split signal 
   mprintf("Splitting signal at t = %f s \n", (splitIndex / sampleRate));  
   if splitIndex < 1 | splitIndex >= length(audio) then
       error("Could not split audio into signal and noise part!");
   end
   
   signal = audio(splitIndex+1:$);
   noise = audio(1:splitIndex); 
endfunction

// Return the power spectral density using the Welch method (= Bartlett spectrogram with overlapping windows).
// Length of the spectrum is windowSize/2+1, in FFT buffer format.
function [spectrum] = PowerSpectrum(signal, windowSize, hopSize, windowType, sampleRate)
    // Compute spectrum (Welch)
     spectrum = pspect(hopSize,windowSize, windowType, signal);    
endfunction

// Return a one side power cross spectrum for 2 signals. Length is windowSize/2+1
function [spectrum] = CrossPowerSpectrum(signal1, signal2, windowSize, hopSize, windowType, sampleRate)
    // Compute spectrum (Welch)
     spectrum = pspect(hopSize,windowSize, windowType, signal1, signal2);     
endfunction

// Truncate an impulse response by determing when the IR starts to have above threshold contributions, and when it stops doing so.
// The first action minimizes delay of the IR, while the second one minimizes any gain that has to be added to IR loaders that do not perform audio output normalization.
function truncatedImpulseResponse = TruncateImpulseResponse(impulseResponse, thresholdDBFS, thresholdHysteresisDBFS, RMSwindowSize, minIRLength)
    IRLength = length(impulseResponse);
    
    // Compute RMS for every point.
    impulseResponseRMS = zeros(IRLength);
    for index = 1:IRLength;
        impulseResponseRMS(index) = AmplitudeToRMS(impulseResponse, index, RMSwindowSize);
    end
    
    // DEBUG ...
     //figure();
     //plot(impulseResponseRMS);
    
    //Compute RMS around maximum peak. This gives a reference for thresholding.
    [referenceRMS, peakIndex] = max(impulseResponseRMS);
    thresholdRMS = abs(DBToAmplitude(thresholdDBFS, referenceRMS));  
    mprintf('Reference RMS for IR at index %d: %f\n', peakIndex, referenceRMS);
    mprintf('Threshold RMS at %d dBFS: %f\n', thresholdDBFS, thresholdRMS);
       
    // Start looking from start until first important peak ...
    // NO need to go over the middle, EVER.
    startIndex = 1; // Value used if not set in loop ...
    for index = 1:IRLength/2;
       // Stop looking if we find an above threshold region ...
       // We get a strange error, as if the numbers are complex (but they aren't!)
       if impulseResponseRMS(index) > thresholdRMS then
           // Go back one step ...
           startIndex = max([index - 1,1]);
           mprintf('RMS has crossed threshold up at %d\n', startIndex);
           break;
       end
    end
    
    // Now continue looking until we find a 'flat' below threshold region again.
    endIndex = IRLength;

    // Only makes sense if a proper start was found and we are above thershold!
    if (impulseResponseRMS(index) > thresholdRMS) then
        // We provide some hysterises by halving the threshold RMS.
        thresholdRMS = abs(DBToAmplitude(thresholdDBFS + thresholdHysteresisDBFS, referenceRMS));  
        for index = (startIndex + minIRLength):IRLength;
           // Stop looking if we find a below threshold region ...
           // We get strange error, as if the numbers are complex (but they aren't!)
           if impulseResponseRMS(index) < thresholdRMS then
               // Go further one step ...
               endIndex = min([index + 1, IRLength]);
               mprintf('RMS has crossed threshold down at %d\n', endIndex);
               break;
           end
        end
    end
       
    // Return the truncated IR.
    truncatedImpulseResponse = impulseResponse(startIndex:endIndex);
endfunction

// Truncate any leading zero or below threshold entries from a min-phase IR.
// If zero is used as a treshold no information is lost.
// Much simpler algorithm than the one for above.
function truncatedImpulseResponse = TruncateMinPhaseImpulseResponse(impulseResponse, threshold)
    IRLength = length(impulseResponse);
    for index = 1:IRLength;       
       // Stop looking if we find an above threshold region ...
       if abs(impulseResponse(index)) > threshold then
           break;
       end
    end
    
    // Return the truncated IR.
    truncatedImpulseResponse = impulseResponse(index:IRLength);
endfunction

// Compute a minimum phase (or delay in time domain) version of the transfer function in FFT buffer format 
// based on the cepstrum.
// This assumes a reasonably smooth spectrum, with NO zero amplitude frequencies!
function minPhaseTransferFunction = MinPhaseTransferFunction(transferFunction, clipdB)        
    minPhaseTransferFunction =  exp( fft( FoldFFTTimeDomainData( ifft( log( ClipdB(transferFunction, clipdB) )))));    
endfunction

// Compute RMS in a window.
// The window size is adjusted according to the boundaries of the signal, so it may grow smaller to windowSize / 2.
function rms = AmplitudeToRMS(signal, index, windowSize)
    rms = sqrt(mean(abs(signal([max(1,index - windowSize/2):min(length(signal),index + windowSize/2)])).^2));
endfunction


// Return the number of decibels between signal and reference for 
// amplitude signals. Reference is 1 if not passed in, useful for peak or single amplitude values.
// Use a reference of 0.707 for RMS over a set of values of at least 1 period (the RMS values of 1 unit peak sinusoidal signal is 0.707). 
function db = AmplitudeToDB(amplitude, reference)
    if argn(2) == 1 then reference = 1; end;
    db = 20 * log10(amplitude/reference);
endfunction

// Return the number of decibels between signal and reference for 
// power signals (square of an amplitude). Reference is 1 if not passed in.
// Use 0.707 for RMS^2 values over at least on period.
function db = PowerToDB(power, reference)
    if argn(2) == 1 then reference = 1; end;
    db = 10 * log10(power/reference);
endfunction

function amplitude = DBToAmplitude(db, reference)
    if argn(2) == 1 then reference = 1; end;
    amplitude = reference * 10 ^ (db/20);
endfunction

function power = DBToPower(db, reference)
    if argn(2) == 1 then reference = 1; end;
    power = reference * 10 ^ (db/10);
endfunction

// Select the audio channel of a mono or stereo source
function audio = GetMonoAudio(inputAudio, channel)
    [n1, n2] = size(inputAudio)
    audio = inputAudio(min([n1,channel]),:);
endfunction

// Generate default options for computing impulse responses which can be modified.
function options = GetImpulseResponseOptions()    
    options = struct('channel', 1, ...
                     'fftWindowSize',8192, ...
                     'fftHopSize', 4096, ...
                     'signalNoiseSplitDBFS', -32, ...
                     'signalNoiseSplitTimeWindow', 1, ...
                     'impulseResponseTimeLengthMS', 80, ...                     
                     'transferFunctionEstimator', 'H0', ... // H0 seems to work best ...
                     'WienerCorrection', 'on', ...
                     'transferFunctionSmoothing', 24, ...
                     'transferFunctionMinimumPhaseTransform', 'on', ...
                     'minimumAudioDBFS', -90, ...
                     'truncateImpulseResponse', 'on', ... 
                     'normalizeImpulseResponse', 'on', ...// leaves this to 'on', scilab and most applications cannot handle values > 0 dBFS (= 1.0 in 32 bit float wav files).
                     'impulseResponseTruncateThresholdDBFS', -32, ...
                     'impulseResponseTruncateThresholdHysteresisDBFS', -12, ...
                     'impulseResponseTruncateWindowSize', 9, ...
                     'plotAudio', 'off', ...
                     'plotTransferFunctionAndImpulseResponses', 'on', ...
                     'plotSpectra', 'on', ...                     
                     'plotCutoffFrequency', 50, ...
                     'audioPlotTime', 10);
endfunction

// Smooth data using a running average of given size.
function smoothData =  Smooth(data, windowSize)
    nrPts = length(data);
    smoothData = zeros(nrPts);
    for i=1:nrPts
       smoothData(i) = mean(data(max(1,i-windowSize/2): min(nrPts, i+windowSize/2)));          
    end
endfunction

// Smooth one sided spectrum using a running averager that uses all 
// data points within a given octave band size.
// Frequencies must be in strict ascending order.
// The result has the same number of data points.
function smoothPsd =  SmoothByOctaveBand(frequency, psd, bandsPerOctave)
    nrPts = length(psd);
    smoothPsd = zeros(nrPts);
    deltaFrequency = frequency(2) - frequency(1);
    fd = OctaveBandEdgeFactor(bandsPerOctave);
    
    for i=1:nrPts
        // Get band edges in frequency. Note that they maybe negative!
        band = [frequency(i)/fd, frequency(i)*fd];
        lowerIndex = find(frequency > min(band))(1);
        higherIndex = find(frequency < max(band))($);         
        if (higherIndex - lowerIndex) > 1 then           
            smoothPsd(i) = mean(psd(lowerIndex:higherIndex));          
        else
            smoothPsd(i) = psd(i);
        end 
    end
endfunction

// Smooth subsample one sided spectrum per octave bands.
// Frequency must be equally spaced!
// Returns band info if required ...
// Not used currently.
function [smoothPsd, bandCenter, bandLower, bandUpper] =  SmoothSubsampleByOctaveBand(frequency, psd, bandsPerOctave)
    nrPts = length(psd);
    deltaFrequency = frequency(2) - frequency(1);
    [bandLower bandCenter bandUpper] = OctaveBands(bandsPerOctave);
    nrSampledPts = length(bandCenter);
    smoothPsd = zeros(1,nrSampledPts);
    for i=1:nrSampledPts
        windowSize = 2*floor(((bandUpper(i) - bandLower(i)) / deltaFrequency) / 2) + 1;        
        centerFrequencyIndex = (bandCenter(i) - frequency(1)) / deltaFrequency + 1;
        if windowSize > 1 then           
            smoothPsd(i) = mean(psd(max(1,centerFrequencyIndex-(windowSize-1)/2): min(nrPts, centerFrequencyIndex+(windowSize-1)/2)));          
        else
            smoothPsd(i) = psd(centerFrequencyIndex);
        end 
    end
endfunction

// Octave band definitions.
function [bandLower, bandCenter, bandUpper] = OctaveBands(bandsPerOctave)
    // Center frequencies relate as 10^(3/[10N]), f(b-1) =  f(b) / (10^(3/10N))
    // Center band is usually 1000 Hz. This is not standard compliant!
     multiplier = 10^(3 / (10* bandsPerOctave));
     select bandsPerOctave
      case 1
       // 1000 Hz is band nr 7 out of 11
         bandCenter = 1000 * multiplier ^([1:11] - 7);
      case 3 
       // 1000 Hz is band nr 19 out of 32
         bandCenter = 1000 * multiplier ^([1:32]-19);
      case 6
       // 1000 Hz is band nr 38 out of 64 ?
         bandCenter =  1000 * multiplier ^((1:64)- 38);
      case 12
         // 1000 Hz is band nr 76 out of 128 ?          
         bandCenter =  1000 * multiplier ^((1:128) - 76);
     case 24
         // 1000 Hz is band nr 152 out of 256 ?
         bandCenter =  1000 * multiplier ^((1:256) - 152);
    end
    fd = OctaveBandEdgeFactor(bandsPerOctave);
    bandUpper = bandCenter * fd;
    bandLower = bandCenter / fd;
endfunction

// Return the edge factor given the required number of bands per octave.
function fd = OctaveBandEdgeFactor(bandsPerOctave)
    fd = 2^(1.0 / (2 * bandsPerOctave));
endfunction

//
// Test signal generation
//

// Generate a sine with given oneSidedFrequency and amplitude
function sine = GenerateSine(lengthSeconds, oneSidedFrequencyHz, amplitude, sampleRate)
    index = [0:lengthSeconds * sampleRate];
    sine  = amplitude * sin(index *  2 * %pi  * oneSidedFrequencyHz / sampleRate);
endfunction

// Generate white gaussian noise with given amplitude.
function noise = GenerateNoise(lengthSeconds, amplitude, sampleRate)
    index = [0:lengthSeconds * sampleRate];
    noise  = amplitude * rand(index, 'normal');
endfunction

//
// FFT utilities. Some by other people, see comments.
//

// Fold FFT buffer format data onto left wing: data[1..N] => data[1 .. N/2+1] , e.g. for plotting
// data[1] => data[1] is the DC component
// 2 * data[n] , n=[2..N/2] => data[n-1]
// data[N/2+1] is the Nyquist frequencies component => data[N/2+1]
// We assume a real spectrum.
function foldedData = FoldFFTFrequencyDomainData(data)
     foldedLength = length(data)/2 + 1;
     foldedData = zeros(foldedLength);

     // Copy DC and Nyquist frequency.
     foldedData(1) = data(1);     
     foldedData(foldedLength) = data(foldedLength);
     
     // Throw away negative frequencies, they are the same as the positive frequencies: just double those.
     foldedData(2:foldedLength) = 2 * data(2:foldedLength); 
endfunction

function [rw] = FoldFFTTimeDomainData(r)
// [rw] = fold(r)
// Fold left wing of vector in "FFT buffer format"
// onto right wing
// J.O. Smith, 1982-2002
// When applied to time data (e.g. an impulse response) it makes
// a non-causal time series causal.
// When applied to an inverse fourier transform of a log-spectrum 
// it converts non-minimum-phase zeros to minimum-phase zeros.
   [m,n] = size(r);
   if m*n ~= m+n-1
     error('fold.m: input must be a vector');
   end
   flipped = 0;
   if (m > n)
     n = m;
     r = r.';
     flipped = 1;
   end
   if n < 3, rw = r; return;
   elseif modulo(n,2)==1,
       nt = (n+1)/2;
       rw = [ r(1), r(2:nt) + conj(r(n:-1:nt+1)), ...
             0*ones(1,n-nt) ];
   else
       nt = n/2;
       rf = [r(2:nt),0];
       rf = rf + conj(r(n:-1:nt+1));
       rw = [ r(1) , rf , 0*ones(1,n-nt-1) ];
   end;

   if flipped
     rw = rw.';
   end
endfunction

// Return a vector with frequences for one-sided folded FFT data.
function frequencies = FoldedFFTFrequencies(fftWindowSize, sampleRate)
    frequencies = [1:fftWindowSize / 2] * sampleRate / fftWindowSize;
endfunction
    
// Return a vector with frequences for shifted FFT data.    
function frequencies = ShiftedFFTFrequencies(fftWindowSize, sampleRate)
    frequencies = [-fftWindowSize / 2:fftWindowSize / 2 - 1] * sampleRate / fftWindowSize;
endfunction
    
function [clipped] = ClipdB(s,cutoff)
// [clipped] = clipdb(s,cutoff)
// Clip magnitude of s at its maximum + cutoff in dB.
// Example: clip(s,-100) makes sure the minimum magnitude
// of s is not more than 100dB below its maximum magnitude.
// If s is zero, nothing is done.

    clipped = s;
    as = abs(s);
    mas = max(as(:));
    if mas==0, return; end
    if cutoff >= 0, return; end
    thresh = mas*10^(cutoff/20); // db to linear
    toosmall = find(as < thresh);
    clipped = s;
    clipped(toosmall) = thresh;
endfunction
