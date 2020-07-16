%% Read continuosly
%not working?

function []= testeai1()
dq = daq("ni");
addinput(dq, "Dev1", "ai1", "Voltage");
% addoutput(dq, "Dev1", "ao0", "Voltage");


d.ScansAvailableFcn = @plotMyData;

start(dq,"Duration",100);

while true 
  pause (0.1)
end
  
    function plotMyData(obj,evt)
    % obj is the DataAcquisition object passed in. evt is not used.
    data = read(obj,obj.ScansAvailableFcnCount,"OutputFormat","Matrix");
    plot(data)
    end

end

