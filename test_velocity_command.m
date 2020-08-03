dq = daq("ni"); %create data acquisition
dq.Rate = 1000; %set the generation scan rate; rate cant be the same as K (check system_identification_fc)
addoutput(dq, "Dev1", "ao1", "Voltage");% adds analog output channel 
addinput(dq, "Dev1", "ai1", "Voltage");% adds analog input channel 

% outScanData =zeros(3000,1);
outScanData =(cat(1,10*ones(10000,1),zeros(1000,1),-1*ones(20000,1),zeros(10,1))); %sin(linspace(0,8*pi,3000)')  %creation of signal
% outScanData =(cat(1,0.5*ones(3000,1),-0.5*ones(3000,1)));
inScanData = readwrite(dq,outScanData);

figure(1)
hold on
plot(outScanData)
plot(inScanData.Dev1_ai1)
hold off

position=cumsum(inScanData.Dev1_ai1)/(dq.Rate)*100;

figure(2)
hold on
plot(position)
ylabel('position(mm)')
%plot(position_trapz)
hold off
