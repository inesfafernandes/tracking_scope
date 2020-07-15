function []=testai2()

dq = daq("ni");
dq.Rate = 8000;
addoutput(dq, "Dev1", "ao0", "Voltage");
addinput(dq, "Dev1", "ai1", "Voltage");

%vectors size
n = dq.Rate;

reps=1;
outputSignal1 = sin(linspace(0,2*pi,n)');
% outputSignal2 = linspace(-1,1,n)'; 
% outputSignal=[outputSignal1 outputSignal2];

dq.ScansRequiredFcn  = @loadmoredata;

dq.ScansAvailableFcn = @plotMyData;

preload(dq, repmat(outputSignal1,2,1))

start(dq, "Continuous")

start(dq,"Duration",100);
      
while true 
  pause (0.1)
end
      
      function []=loadmoredata(obj,evt)
        reps=reps+1;
        outputSignalNew=outputSignal1*0.1*reps;

        write(obj,outputSignalNew)
      end
  
  function plotMyData(obj,evt)
    % obj is the DataAcquisition object passed in. evt is not used.
    data = read(obj,obj.ScansAvailableFcnCount,"OutputFormat","Matrix");
    plot(data); ylim([-3 3]);
    end

  
end