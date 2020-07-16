%% writes signals to the daq interface (as many as we want), but not in a continous form


dq = daq("ni"); %create data acquisition
dq.Rate = 8000; %set the generation scan rate
% adds analog output channels
addoutput(dq, "Dev1", "ao0", "Voltage");
addoutput(dq, "Dev1", "ao1", "Voltage");

%vector size
n = dq.Rate;

%signal creation
outputSignal1 = sin(linspace(0,2*pi,n)');
outputSignal2 = linspace(-1,1,n)';

outputSignal=[outputSignal1 outputSignal2]; %creation of matrix composed by both siganls

while 1
    write(dq, outputSignal); %writes both signals
end