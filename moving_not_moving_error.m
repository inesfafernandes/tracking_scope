read_trajectory=readtable('x_trajectory.csv');
x_trajectory=read_trajectory.Var1/10;
xstar=x_trajectory(14000:17000-1);

load('output_read_position_full.mat')
model_direct_command=inScanDat;
load('output_MPC_T170_predicted_full.mat')
MPC_T170_know_future=output;
load('output_MPC_T250_predicted_full.mat')
MPC_T250_know_future=output;
load('output_MPC_T170_naive_model_full.mat')
MPC_T170_naive_model=output;
load('output_MPC_T250_naive_model_full.mat')
MPC_T250_naive_model=output;

[under_100_not_moving_direct_command, under_100_moving_direct_command, velocity_filt_direct_command]=moving_calculator(x_trajectory,model_direct_command);
[under_100_not_moving_170_predicted, under_100_moving_170_predicted, velocity_filt_170_predicted]=moving_calculator(x_trajectory,MPC_T170_know_future);
[under_100_not_moving_250_predicted, under_100_moving_250_predicted, velocity_filt_250_predicted]=moving_calculator(x_trajectory,MPC_T250_know_future);
[under_100_not_moving_170_naive, under_100_moving_170_naive, velocity_filt_170_naive]=moving_calculator(x_trajectory,MPC_T170_naive_model);
[under_100_not_moving_250_naive, under_100_moving_250_naive, velocity_filt_250_naive]=moving_calculator(x_trajectory,MPC_T250_naive_model);

pie_direct_command_not_moving=[under_100_not_moving_direct_command  length(velocity_filt_direct_command)-under_100_not_moving_direct_command];
pie_direct_command_moving=[under_100_moving_direct_command  length(velocity_filt_direct_command)-under_100_moving_direct_command];
pie_170_predicted_not_moving=[under_100_not_moving_170_predicted  length( velocity_filt_170_predicted)-under_100_not_moving_170_predicted];
pie_170_predicted_moving=[under_100_moving_170_predicted  length(velocity_filt_170_predicted)-under_100_moving_170_predicted];
pie_250_predicted_not_moving=[under_100_not_moving_250_predicted  length( velocity_filt_250_predicted)-under_100_not_moving_250_predicted];
pie_250_predicted_moving=[under_100_moving_250_predicted  length(velocity_filt_250_predicted)-under_100_moving_250_predicted];
pie_170_naive_not_moving=[under_100_not_moving_170_naive  length( velocity_filt_170_naive)-under_100_not_moving_170_naive];
pie_170_naive_moving=[under_100_moving_170_naive  length(velocity_filt_170_naive)-under_100_moving_170_naive];
pie_250_naive_not_moving=[under_100_not_moving_250_naive  length( velocity_filt_250_naive)-under_100_not_moving_250_naive];
pie_250_naive_moving=[under_100_moving_250_naive  length(velocity_filt_250_naive)-under_100_moving_250_naive];


figure(1)
subplot(1,2,1)
labels = {'<100micro','>100micro'};
pie(pie_direct_command_not_moving);
legend(labels)
title('not moving direct command')
subplot(1,2,2)
labels = {'<100micro','>100micro'};
pie(pie_direct_command_moving);
legend(labels)
title('moving direct command')


figure(2)
subplot(1,2,1)
labels = {'<100micro','>100micro'};
pie(pie_170_predicted_not_moving);
legend(labels)
title('not moving T=170 predicted')
subplot(1,2,2)
labels = {'<100micro','>100micro'};
pie(pie_170_predicted_moving);
legend(labels)
title('moving T=170 predicted')

figure(3)
subplot(1,2,1)
labels = {'<100micro','>100micro'};
pie(pie_250_predicted_not_moving);
legend(labels)
title('not moving T=250 predicted')
subplot(1,2,2)
labels = {'<100micro','>100micro'};
pie(pie_250_predicted_moving);
legend(labels)
title('moving T=250 predicted')

figure(4)
subplot(1,2,1)
labels = {'<100micro','>100micro'};
pie(pie_170_naive_not_moving);
legend(labels)
title('not moving T=170 naive')
subplot(1,2,2)
labels = {'<100micro','>100micro'};
pie(pie_170_naive_moving);
legend(labels)
title('moving T=170 naive')

figure(5)
subplot(1,2,1)
labels = {'<100micro','>100micro'};
pie(pie_250_naive_not_moving);
legend(labels)
title('not moving T=250 naive')
subplot(1,2,2)
labels = {'<100micro','>100micro'};
pie(pie_250_naive_moving);
legend(labels)
title('moving T=250 naive')

%% cumulative probability plots

err=x_trajectory-model_direct_command(2:end);%altere it to use the full trajectory and use one that only acounts for the moving part
xi=0:50:750;
a=hist(abs(err)*10000,xi);
bar(xi,cumsum(a)/sum(a),'b')
hold on
[~,~,~,err_moving]=moving_calculator(x_trajectory,model_direct_command);
xi=0:50:750;
a=hist(abs(err_moving)*10000,xi);
bar(xi,cumsum(a)/sum(a),'cyan')


