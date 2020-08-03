
dq = daq("ni"); %create data acquisition
dq.Rate = 700; %set the generation scan rate; rate cant be the same as K (check system_identification_fc)
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 


read_trajectory=readtable('x_trajectory.csv');
x_trajectory=read_trajectory.Var1;

scale=20;
vel_command=(diff(x_trajectory)/(1/dq.Rate))/scale;

lim=10;
%vel_command = min(max(vel_command,-lim),lim);



inScanData = readwrite(dq,vel_command);

figure;
hold on;
plot(inScanData.Dev1_ai1*10)
plot(x_trajectory)



plot(x_trajectory(1:end-1) - (inScanData.Dev1_ai1*10) )

