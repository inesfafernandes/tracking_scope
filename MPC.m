%% MPC
clear all;close all;clc
load('H_noise_new_rate700_smooth_100.mat');
h=H;
T=200; %time window of the future
[lin,col]=size(h);
N=lin; %past time
M=T+N-1; % total window of time
lambda=1e-2; %constant value to minimize variation of voltage
phi=zeros(T,M);
phi(1,:)=cat(1,h(end:-1:1),zeros(T-1,1)); %sliding window of h
phi_a=zeros(T,N-1);
phi_a(1,:)=phi(1,1:N-1); 
phi_b=zeros(T,T);
phi_b=phi(1,N:end);
xstar=sin(linspace(0,6*pi,T)');%sawtooth(linspace(0,8*pi,T)');%1; % position we want to reach 
up=zeros(N-1,1); %u past
I=eye(T); %identity matrix with size 20x20
uf=zeros(1,T); %initializing u future matrix

%building matrix phi, phi_a and phi_b
for i=2:T
    for j=2:M
        phi(i,j)=phi(i-1,j-1);
        phi_a(i,:)=phi(i,1:499);% sliding window of h for past values
        phi_b(i,:)=phi(i,500:end); %sliding window of h for future values
    end
end

uf=inv((lambda*I+phi_b'*phi_b))*phi_b'*(xstar-phi_a*up);
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
hold off

figure(2)
trajectory=conv(u,h,'valid');
plot(trajectory);

    
