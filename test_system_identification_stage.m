%% sends an input to the stage and reads its response; calculates the stage Kernel (H)

dq = daq("ni"); %create data acquisition
dq.Rate = 8000; %set the generation scan rate
addoutput(dq, "Dev1", "ao1", "Voltage"); % adds analog output channel
addinput(dq, "Dev1", "ai1", "Voltage"); % adds analog input channel

%vectors size
n = dq.Rate;
global data;%creates global variable data that stores de input signal being read
data = [];
global reps;%creates global variable reps
reps=1; %initialization of reps
%creation of signals

global outputSignal1; %creates global variable outputSignal1
outputSignal1=randn(n,1);
outputSignal1=min(max(outputSignal1,-1),1);

dq.ScansRequiredFcn  = @loadmoredata; % assign callback function to the ScansRequiredFcn of the daq to continuosly generate the output data

dq.ScansAvailableFcn = @plotMyData; % assign callback function to the ScansAvailableFcn of the daq to continuosly acquire the input data

preload(dq, repmat(outputSignal1,2,1)) % Before starting a continuous generation, preload outputSignal

start(dq, "Continuous") % to initiate the generation in continuos form

start(dq); % to initiate the acquisition

%H=system_identification_fc( outputSignal1, data) %uses function to return kernel H

% while true
%     pause (0.1)
% end

