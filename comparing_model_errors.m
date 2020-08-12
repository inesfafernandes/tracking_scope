%% Comparing model errors

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
% 
% RMSE_direct_command= sqrt(mean((xstar-model_direct_command(2:end)).^2)); %root mean square error
% RMSE_MPC_T20_know_future= sqrt(mean((xstar(20:end)-MPC_T20_know_future).^2));
% RMSE_MPC_T50_know_future= sqrt(mean((xstar(50:end)-MPC_T50_know_future).^2));
% RMSE_MPC_T100_know_future= sqrt(mean((xstar(100:end)-MPC_T100_know_future).^2));
% RMSE_MPC_T20_naive_model= sqrt(mean((xstar-MPC_T20_naive_model(2:end)).^2));
% RMSE_MPC_T50_naive_model= sqrt(mean((xstar-MPC_T50_naive_model(2:end)).^2));
% RMSE_MPC_T100_naive_model= sqrt(mean((xstar-MPC_T100_naive_model(2:end)).^2));

RMSE_direct_command=RMSE_calculator(xstar,model_direct_command);
RMSE_MPC_T20_know_future=RMSE_calculator(xstar,MPC_T20_know_future);
RMSE_MPC_T50_know_future=RMSE_calculator(xstar,MPC_T50_know_future);
RMSE_MPC_T100_know_future=RMSE_calculator(xstar,MPC_T100_know_future);
RMSE_MPC_T20_naive_model=RMSE_calculator(xstar,MPC_T20_naive_model);
RMSE_MPC_T50_naive_model=RMSE_calculator(xstar,MPC_T50_naive_model);
RMSE_MPC_T100_naive_model=RMSE_calculator(xstar,MPC_T100_naive_model);

error_direct_command=xstar-model_direct_command(2:end);
error_MPC_T20_know_future=xstar(1:end-19)-MPC_T20_know_future;
error_MPC_T50_know_future=xstar(1:end-49)-MPC_T50_know_future;
error_MPC_T100_know_future=xstar(1:end-99)-MPC_T100_know_future;
error_MPC_T20_naive_model=xstar-MPC_T20_naive_model(2:end);
error_MPC_T50_naive_model=xstar-MPC_T50_naive_model(2:end);
error_MPC_T100_naive_model=xstar-MPC_T100_naive_model(2:end);

[error_count_direct_command,max_error_direct_command,min_error_direct_command]=error_calculator(error_direct_command);
[error_count_MPC_T20_know_future,max_error_MPC_T20_know_future,min_error_MPC_T20_know_future]=error_calculator(error_MPC_T20_know_future);
[error_count_MPC_T50_know_future,max_error_MPC_T50_know_future,min_error_MPC_T50_know_future]=error_calculator(error_MPC_T50_know_future);
[error_count_MPC_T100_know_future,max_error_MPC_T100_know_future, min_error_MPC_T100_know_future]=error_calculator(error_MPC_T100_know_future);
[error_count_MPC_T20_naive_model, max_error_MPC_T20_naive_model,min_error_MPC_T20_naive_model]=error_calculator(error_MPC_T20_naive_model);
[error_count_MPC_T50_naive_model,max_error_MPC_T50_naive_model,min_error_MPC_T50_naive_model]=error_calculator(error_MPC_T50_naive_model);
[error_count_MPC_T100_naive_model, max_error_MPC_T100_naive_model, min_error_MPC_T100_naive_model]=error_calculator(error_MPC_T100_naive_model);

max_errors=[max_error_direct_command, max_error_MPC_T20_know_future, max_error_MPC_T50_know_future, max_error_MPC_T100_know_future, max_error_MPC_T20_naive_model, max_error_MPC_T50_naive_model,max_error_MPC_T100_naive_model];


figure(1)
hold on
plot(error_direct_command);
plot(error_MPC_T20_know_future);
plot(error_MPC_T50_know_future);
plot(error_MPC_T100_know_future);
plot(error_MPC_T20_naive_model);
plot(error_MPC_T50_naive_model);
plot(error_MPC_T100_naive_model);
legend('error direct command','error MPC T20 know future','error MPC T50 know future','error MPC T100 know future','error MPC T20 naive model','error MPC T50 naive model','error MPC T100 naive model');
hold off

l = cell(1,3);
l{1}='<1/3'; l{2}='>1/3 & <2/3'; l{3}='>2/3' 

figure(2)
subplot(2,4,1)
bar(error_count_direct_command);
title('direct command')
set(gca,'xticklabel', l) 
subplot(2,4,2)
bar(error_count_MPC_T20_know_future);
title('MPC T20 know future')
set(gca,'xticklabel', l) 
subplot(2,4,3)
bar(error_count_MPC_T50_know_future);
set(gca,'xticklabel', l)
title('MPC 520 know future')
subplot(2,4,4)
bar(error_count_MPC_T100_know_future);
set(gca,'xticklabel', l) 
title('MPC T100 know future')
subplot(2,4,5)
bar(error_count_MPC_T20_naive_model);
set(gca,'xticklabel', l) 
title('MPC T20 naive model')
subplot(2,4,6)
bar(error_count_MPC_T50_naive_model);
set(gca,'xticklabel', l) 
title('MPC T50 naive model')
subplot(2,4,7)
bar(error_count_MPC_T100_naive_model);
set(gca,'xticklabel', l) 
title('MPC T100 naive model')

figure(3)
hold on
plot(RMSE_direct_command);
plot(RMSE_MPC_T20_know_future);
plot(RMSE_MPC_T50_know_future);
plot(RMSE_MPC_T100_know_future);
plot(RMSE_MPC_T20_naive_model);
plot(RMSE_MPC_T50_naive_model);
plot(RMSE_MPC_T100_naive_model);
legend('error direct command','error MPC T20 know future','error MPC T50 know future','error MPC T100 know future','error MPC T20 naive model','error MPC T50 naive model','error MPC T100 naive model');
hold off

l2 = cell(1,7);
l2{1}='direct command'; l2{2}='MPC T20 know future'; l2{3}='MPC T50 know future' ;l2{4}='MPC T100 know future' ;l2{5}='MPC T20 naive model'; l2{6}='MPC T50 naive model' ;l2{7}='MPC T100 naive model'

figure(4)
bar(max_errors)
set(gca,'xticklabel', l2) 



     
        