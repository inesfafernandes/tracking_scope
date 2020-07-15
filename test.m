dq = daq("ni");
dq.Rate = 8000;
addoutput(dq, "Dev1", "ao0", "Voltage");
addoutput(dq, "Dev1", "ao1", "Voltage");

%vectors size
n = dq.Rate;

%%
%signal creation
outputSignal1 = sin(linspace(0,2*pi,n)');
outputSignal2 = linspace(-1,1,n)';

outputSignal=[outputSignal1 outputSignal2];

while 1
    write(dq, outputSignal);   
end