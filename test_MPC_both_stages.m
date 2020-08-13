%% Sending and reading commands to both stages resorting to MPC

read_trajectory=readtable('x_trajectory.csv');
load('H_noise_new_rate700_smooth_100.mat');
h= H/sum(H);
N=length(h);

x_trajectory=read_trajectory.Var1/10;
T=200;

dq = daq("ni"); %create data acquisition
dq.Rate = 700; %set the generation scan rate
addinput(dq, "Dev1", "ai0", "Voltage");
addoutput(dq, "Dev1", "ao0", "Voltage");
addinput(dq, "Dev1", "ai1", "Voltage");
addoutput(dq, "Dev1", "ao1", "Voltage");

u_1 = MPC_optimization_fc(x_trajectory,h,T);
u_2 = MPC_optimization_fc(-1*x_trajectory,h,T);

outputSignal=[u_1(N-1:end) u_2(N-1:end)];
inScanData = readwrite(dq,outputSignal);

stage_trajectory_1=inScanData.Dev1_ai0;
stage_trajectory_2=inScanData.Dev1_ai1;

dif=abs(length(stage_trajectory_1)-length(x_trajectory));

figure(1)
subplot(1,2,1)
hold on
plot(u_1(N-1:end))
plot(x_trajectory)
title('stage 1 vs command 1')
hold off
subplot(1,2,2)
hold on
plot(u_2(N-1:end))
plot(-1*x_trajectory)
title('stage 2 vs command 2')
hold off

figure(2)
subplot(1,2,1)
hold on
plot(stage_trajectory_1)
plot(x_trajectory(1:end-dif))
title('stage trajectory 1 vs fish trajectory 1')
legend('stage trajectory','fish trajectory')
hold off
subplot(1,2,2)
hold on
plot(stage_trajectory_2)
plot(-1*x_trajectory(1:end-dif))
title('stage trajectory 2 vs fish trajectory 2')
legend('stage trajectory','fish trajectory')
hold off

figure(3)
subplot(1,2,1)
plot(stage_trajectory_1-x_trajectory(1:end-dif))
title('error stage 1')
subplot(1,2,2)
plot(stage_trajectory_2-(-1*x_trajectory(1:end-dif)))
title('error stage 2')

euc_dist_vet=[];

for i=1:length(stage_trajectory_1)
    euc_dist=sqrt((stage_trajectory_1(i)-x_trajectory(i))^2+(stage_trajectory_2(i)-(-1*x_trajectory(i)))^2);
    euc_dist_vet=cat(1,euc_dist_vet,euc_dist);
end


figure(4)
plot(euc_dist_vet)
title('eucladian distance')