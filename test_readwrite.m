dq = daq("ni");
dq.Rate = 8000;
n=dq.Rate
addoutput(dq, "Dev1", "ao0", "Voltage");
addinput(dq, "Dev1", "ai1", "Voltage");
i=0
while true
    outScanData = sin(linspace(0,2*pi,n)'); % Increase output voltage with each scan.
    inScanData = readwrite(dq,outScanData);
    plot(inScanData.Time,inScanData.Dev1_ai1);
end
