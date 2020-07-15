function []=testao2()

dq = daq("ni");
dq.Rate = 8000;
addoutput(dq, "Dev1", "ao0", "Voltage");
addoutput(dq, "Dev1", "ao1", "Voltage");

%addinput(dq, "Dev1", "ai0", "Voltage");

%vectors size
n = dq.Rate;

reps=1;
outputSignal1 = sin(linspace(0,2*pi,n)');
outputSignal2 = linspace(-1,1,n)'; 
outputSignal=[outputSignal1 outputSignal2];

dq.ScansRequiredFcn  = @loadmoredata;

preload(dq, repmat(outputSignal,2,1))
start(dq, "Continuous")
      
while true 
  pause (0.1)
end
      
      function []=loadmoredata(obj,evt)
        reps=reps+1;
        outputSignalNew=outputSignal*0.1*reps;
        if reps==5
            outputSignalNew=outputSignal*0.5*reps;
        end

        write(obj,outputSignalNew)
      end
  
end
