read_trajectory=readtable('x_trajectory.csv');
x_trajectory=read_trajectory.Var1/10;
xstar=x_trajectory(14000:17000-1);

load('output_read_position.mat')
model_direct_command=inScanDat;
load('output_MPC_T20_predicted.mat')
MPC_T20_know_future=output;
load('output_MPC_T50_predicted.mat')
MPC_T50_know_future=output;
load('output_MPC_T100_predicted.mat')
MPC_T100_know_future=output;
load('output_MPC_T20_naive_model.mat')
MPC_T20_naive_model=output;
load('output_MPC_T50_naive_model.mat')
MPC_T50_naive_model=output;
load('output_MPC_T100_naive_model.mat')
MPC_T100_naive_model=output;

% figure(1)
% plot(x_trajectory);
% 
% 
% velocity=diff(x_trajectory);
% velocity_filt= diff(sgolayfilt(x_trajectory,2,13));

velocity_filt_direct_command= diff(sgolayfilt(model_direct_command,2,13));
error_direct_command=xstar-model_direct_command(2:end);

% figure(2)
% hold on
% plot(abs(velocity),'.');
% plot(abs(velocity_filt),'r');

threshold=0.0005;
not_moving=[];
moving=[];

for i=1:length(velocity_filt_direct_command)
    if velocity_filt_direct_command(i)<threshold
        not_moving=[not_moving i];
    else
        moving=[moving i];
    end
end

under_100_not_moving=0;
under_100_moving=0;

for i=not_moving
    if error_direct_command(i)<=0.01
        under_100_not_moving=under_100_not_moving+1;
    end
end

for i=moving
    if error_direct_command(i)<=0.01
        under_100_moving=under_100_moving+1;
    end
end

pie_direct_command_not_moving=[under_100_not_moving  length(velocity_filt_direct_command)-under_100_not_moving];
pie_direct_command_moving=[under_100_moving  length(velocity_filt_direct_command)-under_100_moving];

figure(1)
subplot(1,2,1)
labels = {'<100micro','>100micro'};
pie(pie_direct_command_not_moving);
legend(labels)
title('not moving')
subplot(1,2,2)
labels = {'<100micro','>100micro'};
pie(pie_direct_command_moving);
legend(labels)
title('moving')
        
