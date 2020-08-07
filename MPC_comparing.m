read_trajectory=readtable('x_trajectory.csv');
x_trajectory=read_trajectory.Var1/10;
fish_trajectory=x_trajectory(14000:16000-1);
load('command_continous.mat');
command_continous=u;
load('command_once.mat');
command_once=u;

load('H_noise_new_rate700_smooth_100.mat');%loading the h file
h=H/sum(H);

% load('trajectory_continous.mat');
% trajectory_continous=trajectory;
% load('trajectory_once.mat');
% trajectory_once=trajectory;
% N=500;

% xstar=fish_trajectory;
% fake_trajectory=cat(1,zeros(N-1,1),xstar);

% figure(1)
% hold on
% plot(fake_trajectory(N:end));
% plot(trajectory_once);
% plot(trajectory_continous);
% legend('xstar','stage trajectory u once','stage trajectory continous optimization');
% title('xstar vs stage trajectory');
% hold off
% 
% figure(2)
% hold on
% plot(fake_trajectory(N:end)-trajectory_once);
% plot(fake_trajectory(N:end)-trajectory_continous);
% legend('error stage trajectory u once','error stage trajectory continous optimization');
% title('MPC error');
% hold off

figure(1)
hold on
plot(command_continous(499:end));
plot(command_once(499:end));
plot(fish_trajectory);
legend('u_continous_MPC','u_once_MPC','fish trajectory')
title('xstar and voltage command');
hold off


trajectory_continous=conv(command_continous,h,'valid');%trajectory of the stage
trajectory_once=conv(command_once,h,'valid');

figure(2)
hold on
plot(fish_trajectory);
plot(trajectory_continous);
plot(trajectory_once);
legend('fish trajectory','stage trajectory continous MPC','stage trajectory once MPC');
title('xstar vs stage trajectory');
hold off

figure(3)
hold on
plot(fish_trajectory-trajectory_continous);
plot(fish_trajectory-trajectory_once);
legend('continous MPC error','once MPC error');
title('error');
hold off
