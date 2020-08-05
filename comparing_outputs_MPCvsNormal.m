%% Comparing MPC output with normal output for a fish trajectory

read_trajectory=readtable('x_trajectory.csv');
load('output.mat');
output_normal=inScanDat(1:end-1);
load('output_MPC.mat');
output_MPC=output;
load('output_lower_error.mat');
output_normal_lower_error=inScanDat(1:end-1);
load('output_MPC_lower_error.mat');
output_MPC_lower_error=output;
load('output_MPC_lambda10-4.mat');
output_MPC_lambda_lower=output;


x_trajectory=read_trajectory.Var1/10;
xstar_1=x_trajectory(14000:17000-1);
xstar_2=x_trajectory(10000:13000-1);

figure(1)
hold on
plot(xstar_1)
plot(output_MPC)
plot(output_normal)
plot(output_MPC_lambda_lower)
legend('fish trajectory','MPC output','Normal output','MPC output lambda 10-4');
title('input vs outputs')
hold off

figure(2)
hold on
plot(xstar_1-output_MPC);
plot(xstar_1-output_normal);
plot(xstar_1-output_MPC_lambda_lower);
legend('MPC error','Normal error','MPC error lambda 10-4');
title('MPC error vs Normal error');
hold off

figure(3)
hold on
plot(xstar_2)
plot(output_MPC_lower_error)
plot(output_normal_lower_error)
legend('fish trajectory','MPC output','Normal output');
title('input vs outputs (lower error)')
hold off

figure(4)
hold on
plot(xstar_2-output_MPC_lower_error);
plot(xstar_2-output_normal_lower_error);
legend('MPC error','Normal error');
title('MPC error vs Normal error');
hold off