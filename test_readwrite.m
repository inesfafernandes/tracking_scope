%% Reads and writes a signal, but not in a continous form

dq = daq("ni"); %create data acquisition
dq.Rate = 8000; %set the generation scan rate
n=dq.Rate;
addoutput(dq, "Dev1", "ao0", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 

while true
    outScanData = sin(linspace(0,2*pi,n)'); %creation of signal
    inScanData = readwrite(dq,outScanData); % writes outScanData to the daq interface output channels, and reads inScanData from the daq interface input channels
    plot(inScanData.Time,inScanData.Dev1_ai1); %plots the signal read
end
