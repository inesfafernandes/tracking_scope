
dq = daq("ni"); %create data acquisition
dq.Rate = 700; %set the generation scan rate; rate cant be the same as K (check system_identification_fc)
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 


read_trajectory=readtable('x_trajectory.csv');
x_trajectory=read_trajectory.Var1;

scale=10;%in the C# app
vel_command=(diff(x_trajectory)/(1/dq.Rate))/scale;

%lim=10;
%vel_command = min(max(vel_command,-lim),lim);
inScanData = readwrite(dq,vel_command);

figure(1);
hold on;
plot(vel_command)
plot(inScanData.Dev1_ai1)
hold off

output_read=((cumsum(inScanData.Dev1_ai1)/(1/dq.Rate))/scale)*10^(-4);
%save('output_read_vrlocity','output_read');

figure(3)
hold on
plot(output_read);
plot(x_trajectory/10);
legend('stage position','fish trajectory')
hold off

error_trajectory=x_trajectory(1:end-1)/10-((cumsum(inScanData.Dev1_ai1)/(1/dq.Rate))/scale)*10^(-4);

figure(2)
plot(error_trajectory)

