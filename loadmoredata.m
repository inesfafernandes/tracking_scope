%% function to write the signal

function []=loadmoredata(obj,evt)
% function to write the signal
global outputSignal1;

outputSignalNew = outputSignal1;% sets signal to be noise betwaeen 1 and -1
%outputSignalNew=repmat(outputSignalNew,1,2); % creates matrix with 2 outputSignalNew to feed into both channels

write(obj,outputSignalNew);
end