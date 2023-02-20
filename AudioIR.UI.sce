// This GUI file is generated by guibuilder version 4.2.1
//////////
f=figure('figure_position',[897,234],'figure_size',[656,542],'auto_resize','on','background',[33],'figure_name','Graphic window number %d','dockable','off','infobar_visible','off','toolbar_visible','off','menubar_visible','off','default_axes','on','visible','off');
//////////
handles.dummy = 0;
handles.loadSourceAudio=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','center','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.0328125,0.9041667,0.309375,0.0625],'Relief','default','SliderStep',[0.01,0.1],'String','Load source audio (pickup)','Style','pushbutton','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','loadSourceAudio','Callback','loadSourceAudio_callback(handles)')
handles.loadTargetAudio=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','center','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.0328125,0.7983334,0.309375,0.0625],'Relief','default','SliderStep',[0.01,0.1],'String','Load target audio (mic)','Style','pushbutton','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','loadTargetAudio','Callback','loadTargetAudio_callback(handles)')
handles.sourceAudioFile=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.396875,0.9104167,0.5875,0.05625],'Relief','default','SliderStep',[0.01,0.1],'String','','Style','text','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','sourceAudioFile','Callback','')
handles.targetAudioFile=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.396875,0.7982955,0.5875,0.05625],'Relief','default','SliderStep',[0.01,0.1],'String',' ','Style','text','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','targetAudioFile','Callback','')
handles.computeIR=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','center','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.0328125,0.69875,0.309375,0.0625],'Relief','default','SliderStep',[0.01,0.1],'String','Compute impulse response','Style','pushbutton','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','computeIR','Callback','computeIR_callback(handles)')
handles.playSourceAudio=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','center','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.0328125,0.5991667,0.309375,0.0625],'Relief','default','SliderStep',[0.01,0.1],'String','Play source audio','Style','pushbutton','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','playSourceAudio','Callback','playSourceAudio_callback(handles)')
handles.playTargetAudio=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','center','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.0328125,0.4995833,0.309375,0.0625],'Relief','default','SliderStep',[0.01,0.1],'String','Play target audio','Style','pushbutton','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','playTargetAudio','Callback','playTargetAudio_callback(handles)')
handles.playComputedAudio=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','center','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.0328125,0.4,0.309375,0.0625],'Relief','default','SliderStep',[0.01,0.1],'String','Play transformed source audio','Style','pushbutton','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','playComputedAudio','Callback','playComputedAudio_callback(handles)')
handles.saveImpulseResponse=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','center','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.0328125,0.2954545,0.309375,0.0625],'Relief','default','SliderStep',[0.01,0.1],'String','Save impulse response','Style','pushbutton','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','saveImpulseResponse','Callback','saveImpulseResponse_callback(handles)')
handles.minimumPhaseTransform=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.3984375,0.7,0.28,0.0590909],'Relief','default','SliderStep',[0.01,0.1],'String','Minimum Phase Transform','Style','checkbox','Value',[1],'VerticalAlignment','middle','Visible','on','Tag','minimumPhaseTransform','Callback','minimumPhaseTransform_callback(handles)')
handles.truncateImpulseResponse=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.3984375,0.6005682,0.28,0.0590909],'Relief','default','SliderStep',[0.01,0.1],'String','Truncate Impulse Response','Style','checkbox','Value',[1],'VerticalAlignment','middle','Visible','on','Tag','truncateImpulseResponse','Callback','truncateImpulseResponse_callback(handles)')
handles.channel=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[1],'Max',[1],'Min',[0],'Position',[0.7015625,0.5,0.28,0.08],'Relief','default','SliderStep',[0.01,0.1],'String','Left|RightIfPresent','Style','listbox','Value',[1],'VerticalAlignment','middle','Visible','on','Tag','channel','Callback','channel_callback(handles)')
handles.wienerCorrection=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.7,0.6,0.28,0.0545455],'Relief','default','SliderStep',[0.01,0.1],'String','Wiener correction','Style','checkbox','Value',[1],'VerticalAlignment','middle','Visible','on','Tag','wienerCorrection','Callback','wienerCorrection_callback(handles)')
handles.transferFunctionSmoothing=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[5],'Max',[1],'Min',[0],'Position',[0.7,0.3,0.2815625,0.08],'Relief','default','SliderStep',[0.01,0.1],'String','0 (no smoothing)|1 band/octave|3 bands/octave|6 bands/octave|12 bands/octave|24 bands/octave','Style','listbox','Value',[6],'VerticalAlignment','middle','Visible','on','Tag','transferFunctionSmoothing','Callback','transferFunctionSmoothing_callback(handles)')
handles.normalizeImpulseResponse=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.7,0.7,0.28,0.0545455],'Relief','default','SliderStep',[0.01,0.1],'String','Normalize Impulse Response','Style','checkbox','Value',[1],'VerticalAlignment','middle','Visible','on','Tag','normalizeImpulseResponse','Callback','normalizeImpulseResponse_callback(handles)')
handles.IRLength=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[200],'Min',[10],'Position',[0.6984375,0.3977272,0.28,0.0545455],'Relief','default','SliderStep',[1,5],'String','IRlength','Style','slider','Value',[80],'VerticalAlignment','middle','Visible','on','Tag','IRLength','Callback','IRLength_callback(handles)')
handles.irLengthLabel=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.3984375,0.4017045,0.28,0.0590909],'Relief','default','SliderStep',[0.01,0.1],'String','Impulse response length (ms)','Style','text','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','irLengthLabel','Callback','')
handles.channelLabel=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.3984375,0.5011364,0.28125,0.0590909],'Relief','default','SliderStep',[0.01,0.1],'String','Channel (Left / Right)','Style','text','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','channelLabel','Callback','')
handles.transferFunctionSmoothingLabel=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.3984375,0.3022727,0.2828125,0.0590909],'Relief','default','SliderStep',[0.01,0.1],'String','Transfer fnct. smoothing','Style','text','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','transferFunctionSmoothingLabel','Callback','')


handles.author=uicontrol(f,'unit','normalized','BackgroundColor',[-1,-1,-1],'Enable','on','FontAngle','normal','FontName','Tahoma','FontSize',[12],'FontUnits','points','FontWeight','normal','ForegroundColor',[-1,-1,-1],'HorizontalAlignment','left','ListboxTop',[],'Max',[1],'Min',[0],'Position',[0.10,0.050,0.8,0.0590909],'Relief','default','SliderStep',[0.01,0.1],'String','ComputeIR 1.0 by Yves Vander Haeghen, dec. 2021, dec. 2022','Style','text','Value',[0],'VerticalAlignment','middle','Visible','on','Tag','channelLabel','Callback','')


f.visible = "on";


//////////
// Callbacks are defined as below. Please do not delete the comments as it will be used in coming version
//////////

// Variables.
global inputData;
inputData = struct('dataPath', '.','sourceAudioFile','', 'targetAudioFile','' );

global outputData;
outputData = struct('impulseResponse', [], 'sourceAudio',[], 'targetAudio',[], 'computedTargetAudio',[], 'sampleRate',1, 'IRGain', 1);

// These should be the same as the default UI settings!!!!!
// Note that currently we do not provide UI access to all the settings ...
global options;
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

function loadSourceAudio_callback(handles)
//Write your callback for  loadSourceAudio  here
    global inputData;
    [inputData.sourceAudioFile inputData.dataPath] = uigetfile(["*.wav";],inputData.dataPath,"Select source audio file");
    set(handles.sourceAudioFile, "String", inputData.sourceAudioFile);
endfunction


function loadTargetAudio_callback(handles)
//Write your callback for  loadTargetAudio  here
    global inputData;
    [inputData.targetAudioFile inputData.dataPath] = uigetfile(["*.wav";],inputData.dataPath,"Select target audio file");
    set(handles.targetAudioFile, "String", inputData.targetAudioFile);
endfunction


function computeIR_callback(handles)
//Write your callback for  computeIR  here
   global inputData;
   global outputData;
   global options;
   exec("audioIR.sci")
   fullSourceFileName  = inputData.dataPath + "/" + inputData.sourceAudioFile;
   fullTargetFileName  = inputData.dataPath + "/" + inputData.targetAudioFile; 
   [outputData.impulseResponse, outputData.sourceAudio, outputData.targetAudio, outputData.computedTargetAudio, outputData.sampleRate] = AudioIR(options, fullSourceFileName, fullTargetFileName);
endfunction


function playSourceAudio_callback(handles)
//Write your callback for  playSourceAudio  here
    global outputData;
    playsnd(outputData.sourceAudio, outputData.sampleRate); 
endfunction


function playTargetAudio_callback(handles)
//Write your callback for  playTargetAudio  here
    global outputData;
    playsnd(outputData.targetAudio, outputData.sampleRate);
endfunction


function playComputedAudio_callback(handles)
//Write your callback for  playComputedAudio  here
  global outputData;
  playsnd(outputData.computedTargetAudio, outputData.sampleRate);
endfunction


function saveImpulseResponse_callback(handles)
//Write your callback for  saveImpulseResponse  here
   global outputData;
   [saveFile path] = uiputfile(["*.wav";],inputData.dataPath,"Save IR (32 bit float)")
    wavwrite(outputData.impulseResponse,outputData.sampleRate , 32, path + '\' + saveFile);
endfunction


function minimumPhaseTransform_callback(handles)
//Write your callback for  minimumPhaseTransform  here
   global options;
   if (handles.minimumPhaseTransform.Value == 1) then
     options.transferFunctionMinimumPhaseTransform = 'on';
   else
     options.transferFunctionMinimumPhaseTransform =  'off';
   end
    mprintf("Options.transferFunctionMinimumPhaseTransform = %s\n", options.transferFunctionMinimumPhaseTransform);
endfunction


function truncateImpulseResponse_callback(handles)
//Write your callback for  truncateImpulseResponse  here
   global options;
   if (handles.truncateImpulseResponse.Value == 1) then
     options.truncateImpulseResponse = 'on';
   else
     options.truncateImpulseResponse =  'off';
   end
    mprintf("Options.truncateImpulseResponse = %s\n", options.truncateImpulseResponse);
endfunction


function channel_callback(handles)
//Write your callback for  channel  here
  global options;
  options.channel = handles.channel.Value;
  mprintf("Options.channel = %d\n", options.channel);
endfunction


function wienerCorrection_callback(handles)
//Write your callback for  wienerCorrection  here
   global options;
   if (handles.wienerCorrection.Value == 1) then
     options.WienerCorrection = 'on';
   else
     options.WienerCorrection =  'off';
   end
    mprintf("Options.WienerCorrection = %s\n", options.WienerCorrection);
endfunction


function transferFunctionSmoothing_callback(handles)
//Write your callback for  tranferFunctionSmoothing  here
   global options;
   select handles.transferFunctionSmoothing.Value
       case 1
            options.transferFunctionSmoothing =  0;   
       case 2
            options.transferFunctionSmoothing =  1;   
       case 3
            options.transferFunctionSmoothing =  3;   
       case 4
            options.transferFunctionSmoothing =  6;   
       case 5
            options.transferFunctionSmoothing =  12;   
       case 6
            options.transferFunctionSmoothing =  24;   
   end
   
    mprintf("Options.transferFunctionSmoothing = %d\n", options.transferFunctionSmoothing);
endfunction


function normalizeImpulseResponse_callback(handles)
//Write your callback for  normalizeImpulseResponse  here
   global options;
   if (handles.normalizeImpulseResponse.Value == 1) then
     options.normalizeImpulseResponse = 'on';
   else
     options.normalizeImpulseResponse =  'off';
   end
    mprintf("Options.normalizeImpulseResponse = %s\n", options.normalizeImpulseResponse);

endfunction


function IRLength_callback(handles)
//Write your callback for  IRLength  here
   global options;
   options.impulseResponseTimeLengthMS = handles.IRLength.Value;
   mprintf("Options.impulseResponseTimeLengthMS = %d ms\n", options.impulseResponseTimeLengthMS);
endfunction


