function []=loadmoredata(obj,evt)
% function to write the signal
global reps;
reps=reps+1;
global outputSignal1;

outputSignalNew=outputSignal1*0.1*reps;

write(obj,outputSignalNew);
end