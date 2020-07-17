%% function to read and plot signal

function [] = plotMyData(obj,evt) %how to acess the variable data?
%function to read and plot signal
% obj is the DataAcquisition object passed in. evt is not used.
dt = read(obj,obj.ScansAvailableFcnCount,"OutputFormat","Matrix");
global data;
data = [data , dt];%concatenates matrix data with array dt

plot(data(:));
end