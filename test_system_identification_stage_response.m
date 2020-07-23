%% Reads and writes a signal, but not in a continous form

dq = daq("ni"); %create data acquisition
dq.Rate = 10000; %set the generation scan rate; rate cant be the same as K (check system_identification_fc)
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 
K=350;

lim=10;

input=zeros(50000,1);
input(50001:100000,1)=1;

outScanData = input; %min(max(5*randn(1e5,1),-lim),lim); %creation of signal
inScanData = readwrite(dq,outScanData); % writes outScanData to the daq interface output channels, and reads inScanData from the daq interface input channels
figure(1)
hold on
plot(outScanData);%title ('input');
plot(inScanData.Dev1_ai1);%title('output'); %plots the signal read
plot(outScanData-inScanData.Dev1_ai1);%title('error');
legend('input','output','error');
hold off


H=system_identification_fc(outScanData, inScanData.Dev1_ai1,K);
%save('H_noise_not_flipped','H');

