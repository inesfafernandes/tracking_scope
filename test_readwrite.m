%% Reads and writes a signal, but not in a continous form

dq = daq("ni"); %create data acquisition
dq.Rate = 1000; %set the generation scan rate; 
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 

lim=10;

outScanData = min(max(5*randn(1e5,1),-lim),lim); %creation of signal
inScanData = readwrite(dq,outScanData); % writes outScanData to the daq interface output channels, and reads inScanData from the daq interface input channels
figure(1)
plot(outScanData);title ('input');
figure(2)
plot(inScanData.Time,inScanData.Dev1_ai1);title('output'); %plots the signal read


