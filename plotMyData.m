function [] = plotMyData(obj,evt) %how to acess the variable data?
%function to read and plot signal
% obj is the DataAcquisition object passed in. evt is not used.
dt = read(obj,obj.ScansAvailableFcnCount,"OutputFormat","Matrix");
global data;
data = [data , dt];


plot(data(:)); ylim([-3 3]);
end