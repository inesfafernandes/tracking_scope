%% Reads and writes a signal, but not in a continous form

dq = daq("ni"); %create data acquisition
dq.Rate = 700; %set the generation scan rate; rate cant be the same as K (check system_identification_fc)
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 
K=500;

lim=10;%10

%input=zeros(50000,1);
%input(50001:100000,1)=1;

outScanData = min(max(randn(1e5,1),-lim),lim); %creation of signal
outScanData(1:100)=0;
outScanData = smooth(outScanData,100); %100

% acc_in2=diff(diff(outScanData2))*(700^2)*10;
% figure
% plot(acc_in2)
inScanData = readwrite(dq,outScanData); % writes outScanData to the daq interface output channels, and reads inScanData from the daq interface input channels
acc=diff(diff(inScanData.Dev1_ai1))*(700^2)*10;
acc_in=diff(diff(outScanData))*(700^2)*10;
figure(1)
hold on
plot(outScanData);%title ('input');
plot(inScanData.Dev1_ai1);%title('output'); %plots the signal read
plot(outScanData-inScanData.Dev1_ai1);%title('error');
legend('input','output','error');
hold off
figure(2)
hold on
plot(acc);
plot(acc_in);
hold off


H=system_identification_fc(outScanData, inScanData.Dev1_ai1,K);
%H=system_identification_fc(acc_in, acc, K);
save('H_velocity','H');
%%
figure(3)
plot(cumsum(H));
%%
in=outScanData;
out_data = inScanData.Dev1_ai1;
out_theory = conv(in,H,'valid');

out_data = out_data(500:end);

figure(2)
hold on
plot(out_data);
plot(out_theory);
