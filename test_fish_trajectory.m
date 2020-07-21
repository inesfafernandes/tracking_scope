%%following fish trajectory_x

read_trajectory=readtable('x_trajectory.csv');
read_H=readtable('H_noise.csv');
x_trajectory=read_trajectory.Var1/10;
H_noise=read_H.Var1;

theoretical_out=conv(x_trajectory,H_noise(end:-1:1),'same');

dq = daq("ni"); %create data acquisition
dq.Rate = 700; %set the generation scan rate; rate cant be the same as K (check system_identification_fc)
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 
K=350;

addzeros=zeros(50,1);

addbegin_trajectory=cat(1,addzeros,x_trajectory);

outScanData = x_trajectory;  %creation of signal
inScanData = readwrite(dq,outScanData); % writes outScanData to the daq interface output channels, and reads inScanData from the daq interface input channels
exp_error=outScanData-inScanData.Dev1_ai1;
theo_error=x_trajectory-theoretical_out;
figure (1)
hold on
plot(outScanData);title ('input vs outputs');
plot(inScanData.Dev1_ai1); %plots the signal read;
plot(theoretical_out);
legend('input','output read','output computed');
hold off
figure(2)
hold on
plot(exp_error);title('error');%title('experimental vs theoretical error');
plot(theo_error);
legend('experimental','theoretical');
hold off

%H=system_identification_fc(outScanData, inScanData.Dev1_ai1,K);
