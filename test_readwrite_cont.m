%% reads and writes continously

function []=testai2()

dq = daq("ni"); %create data acquisition
dq.Rate = 8000; %set the generation scan rate
addoutput(dq, "Dev1", "ao0", "Voltage"); % adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage"); % adds analog input channel 

%vectors size
n = dq.Rate;

reps=1; %initialization of reps
%creation of signals
outputSignal1 = sin(linspace(0,2*pi,n)');
% outputSignal2 = linspace(-1,1,n)'; 
% outputSignal=[outputSignal1 outputSignal2];

dq.ScansRequiredFcn  = @loadmoredata; % assign callback function to the ScansRequiredFcn of the daq to continuosly generate the output data

dq.ScansAvailableFcn = @plotMyData; % assign callback function to the ScansAvailableFcn of the daq to continuosly acquire the input data

preload(dq, repmat(outputSignal1,2,1)) % Before starting a continuous generation, preload outputSignal

start(dq, "Continuous") % to initiate the generation in continuos form

start(dq); % to initiate the acquisition
i=0;

while i<50
  i=i+1;
  pause (0.1)
end


      % function to write the signal
      function []=loadmoredata(obj,evt)
        reps=reps+1;
        outputSignalNew=outputSignal1*0.1*reps;

        write(obj,outputSignalNew);
      end
  
  %function to read and plot signal
  function data = plotMyData(obj,evt) %how to acess the variable data
    % obj is the DataAcquisition object passed in. evt is not used.
    data = read(obj,obj.ScansAvailableFcnCount,"OutputFormat","Matrix");
    plot(data); ylim([-3 3]);
  end

% [data]= plotMyData(obj,evt);
  
end