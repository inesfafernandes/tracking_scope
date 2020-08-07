%% MPC continous optimization

read_trajectory=readtable('x_trajectory.csv');
x_trajectory=read_trajectory.Var1/10;% since the fish trajectory is in mm we need to convert it to cm
fish_trajectory=x_trajectory(14000:16000-1);%we take only a portion of the fish trajectory (in this case the part which has the biggest error)

load('H_smaller.mat');%loading the h file
h=H/sum(H);
T=20; %time window of the future
[lin,~]=size(h);
N=lin; %past time (which is the size of vector h)
M=T+N-1; % total window of time
lambda=1e-4; %constant value to minimize variation of voltage
phi=zeros(T,M);
phi(1,:)=cat(1,h(end:-1:1),zeros(T-1,1)); %sliding window of h
phi_a=zeros(T,N-1);
phi_a(1,:)=phi(1,1:N-1);% first row of phi_a
phi_b=zeros(T,T);
phi_b=phi(1,N:end);% first row of phi_b
%xstar=fish_trajectory; % trajectory we want to follow
up=zeros(N-1,1); %u past
I=eye(T); %identity matrix with size 20x20
uf=zeros(1,T); %initializing u future matrix
model=ones(T,1);%model for predicting where the fish will be in T time steps (this is assuming he will stay where he is at the moment)
u=zeros(N-1,1);%vector that is going to save all commands sent
%building matrix phi, phi_a and phi_b

for i=2:T
    for j=2:M
        phi(i,j)=phi(i-1,j-1);
    end
end

phi_a(2:T,:)=phi(2:T,1:N-1);% sliding window of h for past values
phi_b(2:T,:)=phi(2:T,N:end); %sliding window of h for future values

const_up=((lambda*I+phi_b'*phi_b)\phi_b');% pre computing the part which is constant, to speed up the process

for t=1:length(fish_trajectory)
    uf=const_up*((fish_trajectory(t)*model)-phi_a*up);%computing u future x axis
    %uf_y=const_up*((fish_trajectory_y(t)*model)-phi_a*up); %computing u future y axis
    up=cat(1,up,uf(1));% updating u past by adding the first element of uf as the last element of u past
    up(1)=[];% and discarding the first value of u past
    u=cat(1,u,uf(1));%vector that contains all commands that were sent
end

%taking into consideration tail movement
% for t=1:length(fish_trajectory)
%     if theta<= m
%         uf=const_up*((fish_trajectory(t)*model_m)-phi_a*up);%computing u future x axis
%         %uf_y=const_up*((fish_trajectory_y(t)*model_x)-phi_a*up); %computing u future y axis
%         up=cat(1,up,uf(1));% updating u past by adding the first element of uf as the last element of u past
%         up(1)=[];% and discarding the first value of u past
%         u=cat(1,u,uf(1));%vector that contains all commands that were sent
%     else
%         uf=const_up*((fish_trajectory(t)*model_n)-phi_a*up);%computing u future x axis
%         %uf_y=const_up*((fish_trajectory_y(t)*model_n)-phi_a*up); %computing u future y axis
%         up=cat(1,up,uf(1));% updating u past by adding the first element of uf as the last element of u past
%         up(1)=[];% and discarding the first value of u past
%         u=cat(1,u,uf(1));%vector that contains all commands that were sent
%     end
% end

figure(1)
hold on
plot(u(N-1:end));
plot(fish_trajectory);
legend('u','xstar')
title('xstar and voltage command');
hold off

figure(2)
trajectory=conv(u,h,'valid');%trajectory of the stage
plot(trajectory);
title('stage trajectory');

save ('command_continous','u');

figure(3)
hold on
plot(fish_trajectory);
plot(trajectory);
legend('xstar','stage trajectory');
title('xstar vs stage trajectory');
hold off

figure(4)
plot(fish_trajectory-trajectory);
title('error');


    
