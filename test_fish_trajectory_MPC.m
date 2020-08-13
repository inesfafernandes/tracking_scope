read_trajectory=readtable('x_trajectory.csv');
load('H_noise_new_rate700_smooth_100.mat');
h= H/sum(H);
N=length(h);

x_trajectory=read_trajectory.Var1/10;
xstar=x_trajectory(14000:17000-1);%cat(1,zeros(1000,1),ones(2000,1));%sin(linspace(0,6*pi,3000)');
T=170;

dq = daq("ni"); %create data acquisition
dq.Rate = 700; %set the generation scan rate; rate cant be the same as K (check system_identification_fc)
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 

%uf = MPC_fc(xstar,h,T);
u = MPC_optimization_fc(x_trajectory,h,T);

outScanData = u(N-1:end);  %creation of signal
inScanData = readwrite(dq,outScanData); % writes outScanData to the daq interface output channels, and reads inScanData from the daq interface input channels
output=inScanData.Dev1_ai1;
%save('output_MPC_T100_predicted','output');

figure(1)
hold on 
plot(inScanData.Dev1_ai1);
plot(x_trajectory);
hold off

dif=abs(length(inScanData.Dev1_ai1)-length(x_trajectory));

%%
figure(2)
plot(inScanData.Dev1_ai1-x_trajectory(1:end-dif));




