%% MPC
read_trajectory=readtable('x_trajectory.csv');
x_trajectory=read_trajectory.Var1/10;
fish_trajectory=x_trajectory(14000:17000-1);

load('H_noise_new_rate700_smooth_100.mat');
h=H/sum(H);
T=3000; %time window of the future
[lin,~]=size(h);
N=lin; %past time
M=T+N-1; % total window of time
lambda=1e-4; %constant value to minimize variation of voltage
phi=zeros(T,M);
phi(1,:)=cat(1,h(end:-1:1),zeros(T-1,1)); %sliding window of h
phi_a=zeros(T,N-1);
phi_a(1,:)=phi(1,1:N-1); 
phi_b=zeros(T,T);
phi_b=phi(1,N:end);
xstar=fish_trajectory;%sin(linspace(0,6*pi,T)');%sawtooth(linspace(0,8*pi,T)');%1; % position we want to reach 
up=zeros(N-1,1); %u past
I=eye(T); %identity matrix with size 20x20
uf=zeros(1,T); %initializing u future matrix

%building matrix phi, phi_a and phi_b

for i=2:T
    for j=2:M
        phi(i,j)=phi(i-1,j-1);
    end
end

phi_a(2:T,:)=phi(2:T,1:499);% sliding window of h for past values
phi_b(2:T,:)=phi(2:T,500:end); %sliding window of h for future values

uf=((lambda*I+phi_b'*phi_b)\phi_b')*(xstar-phi_a*up);

%for when we want to actualize up using the first element of uf because the
%fish is constantly moving
% uf=inv((lambda*I+phi_b'*phi_b))*phi_b'*(xstar-phi_a*up);
% up=cat(1,up,uf(1));
% up(1)=[];


u=cat(1,up,uf); %array of commands u containing past and future
%heaviside=cat(2,zeros(1,N-1),ones(1,T));% response we are trying to follow
fake_trajectory=cat(1,zeros(N-1,1),xstar);

figure(1)
hold on
plot(u);
plot(fake_trajectory);
legend('u','xstar')
title('xstar and voltage command');
hold off

figure(2)
trajectory=conv(u,h,'valid');
plot(trajectory);
title('stage trajectory');

figure(3)
hold on
plot(fake_trajectory(500:end));
plot(trajectory);
legend('xstar','stage trajectory');
title('xstar vs stage trajectory');
hold off




    
