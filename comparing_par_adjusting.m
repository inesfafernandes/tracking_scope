load('H_noise_new_rate700_smooth_100.mat');
h1=H;
load('H_noise_new_rate700_smooth_100_testing_par1.mat');
h2=H;
load('H_noise_new_rate700_smooth_100_testing_par2.mat');
h3=H;
load('H_noise_new_rate700_smooth_100_testing_par3.mat');
h4=H;
load('H_noise_new_rate700_smooth_100_testing_par4.mat')
h5=H;
load('H_noise_new_rate700_smooth_100_testing_par5.mat')
h6=H;
load('H_noise_new_rate700_smooth_100_testing_par6.mat')
h7=H;
load('H_noise_new_rate700_smooth_100_testing_par7.mat')
h8=H;
load('H_noise_new_rate700_smooth_100_testing_par8.mat')
h9=H;
load('H_noise_new_rate700_smooth_100_testing_par12.mat')
h10=H;
figure(1)
hold on
plot(h1);
plot(h2);
plot(h3);
plot(h4);
plot(h5);
plot(h6);
plot(h7);
plot(h8);
plot(h9);
plot(h10);
legend('high robustness','par1','par2','par3','par4','par5','par6','par7','par8','par9');
hold off

figure(2)
hold on
plot(cumsum(h1));
plot(cumsum(h2));
plot(cumsum(h3));
plot(cumsum(h4));
plot(cumsum(h5));
plot(cumsum(h6));
plot(cumsum(h7));
plot(cumsum(h8));
plot(cumsum(h9));
plot(cumsum(h10));
legend('high robustness','par1','par2','par3','par4','par5','par6','par7','par8','par9');
hold off