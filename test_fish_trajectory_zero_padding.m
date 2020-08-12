%%following fish trajectory_x with initial zero padding

read_trajectory=readtable('x_trajectory.csv');
load('H_noise_new_rate700_smooth_100.mat');
H_noise_hr = H/sum(H);
% load('H_noise_new_rate700_smooth_100_testing_par6.mat');
% H_noise_par6 = H/sum(H);
% load('H_noise_new_rate700_smooth_100_testing_par8.mat');
% H_noise_par8 = H/sum(H);
% load('H_noise_new_rate700_smooth_100_testing_high-robustness_acc.mat');
% H_noise_hr_acc=H/sum(H);

x_trajectory=read_trajectory.Var1/10;


%scale=10;
p=length(H);
addzeros=zeros(p,1);
addbegin_trajectory=cat(1,addzeros,x_trajectory(14000:17000-1));
%vel_command=(diff(addbegin_trajectory)/(1/dq.Rate))/scale;

%theoretical_out=conv(x_trajectory,H_flipped,'same');
theoretical_out_hr=conv(addbegin_trajectory,H_noise_hr,'valid');
% theoretical_out_hr_acc=conv(addbegin_trajectory,H_noise_hr_acc,'valid');
% theoretical_out_par6=conv(addbegin_trajectory,H_noise_par6,'valid');
% theoretical_out_par8=conv(addbegin_trajectory,H_noise_par8,'valid');

dq = daq("ni"); %create data acquisition
dq.Rate = 700; %set the generation scan rate; rate cant be the same as K (check system_identification_fc)
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 
%K=350;

outScanData = addbegin_trajectory;  %creation of signal
%outScanData=vel_command;
inScanData = readwrite(dq,outScanData); % writes outScanData to the daq interface output channels, and reads inScanData from the daq interface input channels
outScanData_cut = outScanData(p:end);
inScanDat=inScanData.Dev1_ai1(p:end);
save('output_read_position','inScanDat');
theo_error_hr=outScanData_cut-theoretical_out_hr;
% theo_error_hr_acc=outScanData_cut-theoretical_out_hr_acc;
% theo_error_par6=outScanData_cut-theoretical_out_par6;
% theo_error_par8=outScanData_cut-theoretical_out_par8;
exp_error=outScanData_cut-inScanDat;

figure (1)
hold on
plot(outScanData_cut);title ('input vs outputs');
plot(inScanDat); %plots the signal read;
plot(theoretical_out_hr);
%plot(theoretical_out_par6);
%plot(theoretical_out_par8);
legend('input','output read','output computed (high robustness)','output computed (par6)','output computed (par8)');
hold off
figure(2)
hold on
plot(exp_error);title('error');%title('experimental vs theoretical error');
plot(theo_error_hr);
%plot(theo_error_hr_acc);
%plot(theo_error_par6);
%plot(theo_error_par8);
legend('experimental','theoretical_hr','theoretical_hr_acc','theoretical_par6', 'theoretical_par8');
hold off

figure(3)
hold on
plot(outScanData_cut);title ('input vs outputs');
plot(inScanDat); %plots the signal read;
plot(theoretical_out_hr);
%plot(theoretical_out_hr_acc);
legend('input','output read','output computed (high robustness)','output computed (high robustness_acc)');
hold off

%H=system_identification_fc(outScanData, inScanData.Dev1_ai1,K);
