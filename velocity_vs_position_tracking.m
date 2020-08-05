load('output_read_vrlocity.mat');
output_velocity=output_read;
load('output_read_position.mat');
output_position=inScanDat(1:end-2);
read_trajectory=readtable('x_trajectory.csv');
x_trajectory=read_trajectory.Var1;

error_velocity=(x_trajectory(1:end-1)/10)-output_velocity;
error_position=(x_trajectory(1:end-1)/10)-output_position;

figure(1)
hold on
plot(x_trajectory(1:end-1)/10);
plot(output_velocity);
plot(output_position);
legend('fish trajectory','velocity command','position command');
hold off

figure(2)
hold on
plot(error_velocity);
plot(error_position);
plot(detrend(error_velocity));
legend('velocity command error','position command error');
hold off


