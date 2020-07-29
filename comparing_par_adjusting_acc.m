%% comparing H and cumsum of H for different PID parameters, using acc to compute the H

load('H_noise_new_rate700_smooth_100_testing_high-robustness_acc.mat');
h1_a=H;
load('H_noise_new_rate700_smooth_100.mat');
h1=H;
load('H_noise_new_rate700_smooth_100_testing_par1_acc.mat');
h2_a=H;
load('H_noise_new_rate700_smooth_100_testing_par1.mat');
h2=H;
load('H_noise_new_rate700_smooth_100_testing_par2_acc.mat');
h3_a=H;
load('H_noise_new_rate700_smooth_100_testing_par2.mat');
h3=H;
load('H_noise_new_rate700_smooth_100_testing_par1-5_acc.mat');
h4_a=H;
load('H_noise_new_rate700_smooth_100_testing_par1-75_acc.mat');
h5_a=H;
load('H_noise_new_rate700_smooth_100_testing_par1-85_acc.mat');
h6_a=H;
load('H_noise_new_rate700_smooth_100_testing_par1-8_acc.mat');
h7_a=H;

figure(1)
hold on 
plot(h1_a);
plot(h2_a);
plot(h3_a);
legend('high robustness','par1','par2');
hold off

figure(2)
hold on
plot(cumsum(h1_a));
plot(cumsum(h2_a));
plot(cumsum(h4_a));
plot(cumsum(h5_a));
plot(cumsum(h6_a));
plot(cumsum(h7_a));
plot(cumsum(h3_a));
legend('high robustness','par1','par1.5','par1.75','par1.8','par1.85','par2');
hold off

figure(3)
hold on 
plot(cumsum(h1_a));
plot(cumsum(h1),'--');
plot(cumsum(h2_a));
plot(cumsum(h2),'--');
plot(cumsum(h3_a));
plot(cumsum(h3),'--');
legend('high robustness acc', 'high robustness','par2-acc','par2','par3-acc','par3');
hold off

figure(4)
hold on
plot(h1_a);
plot(h1,'--');
plot(h2_a);
plot(h2,'--');
plot(h3_a);
plot(h3,'--');
legend('high robustness acc', 'high robustness','par2-acc','par2','par3-acc','par3');
hold off
