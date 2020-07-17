%% reads and writes continously

% function []=testai2()

dq = daq("ni"); %create data acquisition
dq.Rate = 8000; %set the generation scan rate
addoutput(dq, "Dev1", "ao0", "Voltage"); % adds analog output channel
addinput(dq, "Dev1", "ai1", "Voltage"); % adds analog input channel

%vectors size
n = dq.Rate;
global data;%creates global variable data that stores de input signal being read
data = [];
global reps;%creates global variable reps
reps=1; %initialization of reps
%creation of signals

global outputSignal1; %creates global variable outputSignal1
outputSignal1 = sin(linspace(0,2*pi,n)');
% outputSignal2 = linspace(-1,1,n)';
% outputSignal=[outputSignal1 outputSignal2];

dq.ScansRequiredFcn  = @loadmoredata; % assign callback function to the ScansRequiredFcn of the daq to continuosly generate the output data

dq.ScansAvailableFcn = @plotMyData; % assign callback function to the ScansAvailableFcn of the daq to continuosly acquire the input data

preload(dq, repmat(outputSignal1,2,1)) % Before starting a continuous generation, preload outputSignal

start(dq, "Continuous") % to initiate the generation in continuos form

start(dq); % to initiate the acquisition

while true
    pause (0.1)
end






% [data]= plotMyData(obj,evt);

% end