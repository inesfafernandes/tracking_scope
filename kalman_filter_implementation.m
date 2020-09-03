%% Kalman filter
%https://www.datasciencecentral.com/profiles/blogs/using-bayesian-kalman-filter-to-predict-positions-of-moving
% tutorial (https://www.youtube.com/watch?v=9Ilb4_DJB_s)

%% representation  of the state-space model
%dx/dt=Ax(t)+Bu(t)
%y= Cx(t)
A=[0 1; 0 0];
B=[1 0; 0 0];
C=[1 0];
sys= ss(A,B,C,0);

% building the kalman filter using matlab function

%covariance matrixes
%file:///C:/Users/TeachinglabA/Downloads/Discrete-timeKalmanfilter.pdf
%qn=E(wk.wk'), let us assume [1 0 ; 0 1], wk is the system noise vector
%rn=E(vk^2), let us assume 1, vk is the  measurement noise vector
qn=0.001;
rn=0.001;
%KEST generates optimal estimates y_e(t) and x_e(t)
%L is the estimator gain
%P is the steady state error covariance
[KEST,L,P]=kalman(sys,qn,rn);

%% implementing kalman filter example of tutorial
%from http://www.iri.upc.edu/people/jsola/JoanSola/eng/course.html

%define system
%
% x=[px py vx vy]'
% y=[d,a]'
%
% u=[ax, ay]'
% n=[nx, ny]'
% r=[rd, ra]'
%
% px+=px+vx*dt
% py+=py+vy*dt
% vx+=vx+ax*dt+nx
% vy+=vy+ay*dt+ny
%
% d=sqrt(px^2+py^2)+rd
% a=atan2(py,p'x)+ra
%


dt=1;

F_x=1;
F_u=dt;
F_n=1;
H=1;

Q=0.01;  %covariance of the perturbation
R=100; %(10 m error but its squared)

%simulated variables

X=7; %initial state of x
u=1; %initial state of u, this way it is constant but it could be time dependent

%estimated variables
x=0; %initial estimate
P=1e4; %how much uncertanty

%trajectories
tt=0:dt:100; %time
XX=zeros(1,size(tt,2));
xx=zeros(1,size(tt,2)); %iniatilizing vector of state positions
yy=zeros(1,size(tt,2));
PP=zeros(1,size(tt,2));

%perturbation levels
q=sqrt(Q);
r=sqrt(R);

%start loop(there will be one prediction and one correction on every loop)
i=1;
for t=tt
    %simulate, this part is just to generate y in a real system we have y
    %we dont need this part of the script
    n=q*randn; %perturbation of the system, q is the standard deviation of the noise
    X=F_x*X+F_u*u+F_n*n;
    v=r*randn;
    y=H*X+v;
    
    %estimate
    x=F_x*x+F_u*u; %we dont have acess to n cause we dont know the simulation
    P=F_x*P*F_x'+F_n*Q*F_n';
    
    %correction
    e=y-H*x;
    E= H*P*H';
    
    z=y-e;
    Z=R+E;
    
    K=P*H'*Z^-1; %kalman gain which is being optimazed
    
    x=x+K*z;
    P=P-K*H*P;
    
    %collect data
    XX(:,i)=X;
    xx(:,i)=x;
    yy(:,i)=y;
    PP(:,i)=diag(P);
    
    %update index
    i=i+1;
    
end

plot(tt,XX,tt,xx,tt,yy,tt,sqrt(PP));
legend('truth','state','measurement','sigma');

%%
%file:///C:/Users/TeachinglabA/Downloads/Discrete-timeKalmanfilter.pdf

delT = 1;
F = [ 1 delT; 0 1 ];
H = [ 1 0 ];
x = [ 0 10];
P = [ 10 0; 0 10 ];
Q = [ 1 1; 1 1 ];
R = [ 1 ];
z = [2.5 1 4 2.5 5.5 3 2.3 2.1 5 3.2 4];

state=[]
for i=1:length(z)
    xpred = F*x';
    Ppred = F*P*F' + Q;
    nu = z(i) - H*xpred; 
    S = R + H*Ppred*H';
    K = Ppred*H'*inv(S); %% Kalman gain
    xnew = xpred + K*nu; %% new state
    Pnew = Ppred - K*S*K'; %% new covariance
    state=[state xnew];
end

t=1:length(z);
plot(t,state(1,:),t,z)

%% Matlab kalman filter tracking function
%from https://www.mathworks.com/help/driving/ref/trackingkf.html

%initial state estimate
x = 0;
y = 0;
initialState = [x;0;y;0]; %velocity 0 vector of x position, x velocity, y position, y velocity
KF = trackingKF('MotionModel','2D Constant Velocity','State',initialState); %discrete-time linear Kalman filter used to track the positions and velocities of objects

%real trajectory
vx = 0.2; %velocity on the x axis
vy = 0.1; % velocity on the y axis
T  = 0.5; % time step
pos = [0:vx*T:2;5:vy*T:6]'; %vector position

%Predicting and correcting the state of the object
for k = 1:size(pos,1)
    pstates(k,:) = predict(KF,T);
    cstates(k,:) = correct(KF,pos(k,:));
end

%plotting

plot(pos(:,1),pos(:,2),'k.', pstates(:,1),pstates(:,3),'+',cstates(:,1),cstates(:,3),'o')
xlabel('x [m]')
ylabel('y [m]')
grid
xt  = [x-2 pos(1,1)+0.1 pos(end,1)+0.1];
yt = [y pos(1,2) pos(end,2)];
text(xt,yt,{'First measurement','First position','Last position'})
legend('Object position', 'Predicted position', 'Corrected position')

%other option to check

%% Y. Kim and H. Bang, Introduction to Kalman Filter and Its Applications, InTechOpen, 2018
% Example 2.3 - INS/GNSS navigation
close all
clc
clear

% settings
N = 20; % number of time steps
dt = 1; % time between time steps
M = 100; % number of Monte-Carlo runs
sig_acc_true = [0.3; 0.3; 0.3]; % true value of standard deviation of accelerometer noise
sig_gps_true = [3; 3; 3; 0.03; 0.03; 0.03]; % true value of standard deviation of GPS noise
sig_acc = [0.3; 0.3; 0.3]; % user input of standard deviation of accelerometer noise
sig_gps = [3; 3; 3; 0.03; 0.03; 0.03]; % user input of standard deviation of GPS noise
Q = [diag(0.25*dt^4*sig_acc.^2), zeros(3); zeros(3), diag(dt^2*sig_acc.^2)]; % process noise covariance matrix
R = [diag(sig_gps(1:3).^2), zeros(3); zeros(3), diag(sig_gps(4:6).^2)]; % measurement noise covariance matrix
F = [eye(3), eye(3)*dt; zeros(3), eye(3)]; % state transition matrix
B = [0.5*eye(3)*dt^2; eye(3)*dt]; % control-input matrix
H = eye(6); % measurement matrix

% true trajectory
x_true = zeros(6,N+1); % true state
a_true = zeros(3,N);   % true acceleration
x_true(:,1) = [0; 0; 0; 5; 5; 0]; % initial true state
for k = 2:1:N+1
    x_true(:,k) = F*x_true(:,k-1) + B*a_true(:,k-1);
end

% Kalman filter simulation
res_x_est = zeros(6,N+1,M); % Monte-Carlo estimates
res_x_err = zeros(6,N+1,M); % Monte-Carlo estimate errors
P_diag = zeros(6,N+1); % diagonal term of error covariance matrix

% filtering
for m = 1:1:M
    % initial guess
    x_est(:,1) = [5; -3; 0; 1; 3.1; 0.1];
    P = [eye(3)*4^2, zeros(3); zeros(3), eye(3)*0.4^2];
    P_diag(:,1) = diag(P);
    for k = 2:1:N+1
        
        %%% Prediction
        % obtain acceleration output
        u = a_true(:,k-1) + normrnd(0, sig_acc_true);
        
        % predicted state estimate
        x_est(:,k) = F*x_est(:,k-1) + B*u;
        
        % predicted error covariance
        P = F*P*F' + Q;
        
        %%% Update
        % obtain measurement
        z = x_true(:,k) + normrnd(0, sig_gps_true);
        
        % measurement residual
        y = z - H*x_est(:,k);
        
        % Kalman gain
        K = P\H'/(R+H*P\H');
        
        % updated state estimate
        x_est(:,k) = x_est(:,k) + K*y;
        
        % updated error covariance
        P = (eye(6) - K*H)*P;
        
        P_diag(:,k) = diag(P);
    end
    
    res_x_est(:,:,m) = x_est;
    res_x_err(:,:,m) = x_est - x_true;
    
end

% get result statistics
x_est_avg = mean(res_x_est,3);
x_err_avg = mean(res_x_err,3);
x_RMSE = zeros(3,N+1); % root mean square error
for k = 1:1:N+1
    x_RMSE(1,k) = sqrt(mean(res_x_err(1,k,:).^2,3));
    x_RMSE(2,k) = sqrt(mean(res_x_err(2,k,:).^2,3));
    x_RMSE(3,k) = sqrt(mean(res_x_err(3,k,:).^2,3));
    x_RMSE(4,k) = sqrt(mean(res_x_err(4,k,:).^2,3));
    x_RMSE(5,k) = sqrt(mean(res_x_err(5,k,:).^2,3));
    x_RMSE(6,k) = sqrt(mean(res_x_err(6,k,:).^2,3));
end

% plot results
time = (0:1:N)*dt;
figure
subplot(2,1,1); hold on;
plot(time, x_true(1,:), 'linewidth', 2);
plot(time, res_x_est(1,:,1), '--', 'linewidth', 2);
legend({'True', 'Estimated'}, 'fontsize', 12);
ylabel('X position', 'fontsize', 12); grid on;
subplot(2,1,2); hold on;
plot(time, x_true(4,:), 'linewidth', 2);
plot(time, res_x_est(4,:,1), '--', 'linewidth', 2);
ylabel('X velocity', 'fontsize', 12); xlabel('Time', 'fontsize', 12); grid on;
figure
subplot(2,1,1); hold on;
plot(time, x_RMSE(1,:), 'linewidth', 2);
plot(time, sqrt(P_diag(1,:)), '--', 'linewidth', 2);
legend({'RMSE', 'Estimated'}, 'fontsize', 12);
ylabel('X position error std', 'fontsize', 12); grid on;
subplot(2,1,2); hold on;
plot(time, x_RMSE(4,:), 'linewidth', 2);
plot(time, sqrt(P_diag(4,:)), '--', 'linewidth', 2);
ylabel('X velocity error std', 'fontsize', 12); xlabel('Time', 'fontsize', 12); grid on;
figure
subplot(6,1,1); hold on;
plot(time, res_x_err(1,:,1));
plot(time, res_x_err(1,:,2));
ylabel('p_x', 'fontsize', 12); grid on;
subplot(6,1,2); hold on;
plot(time, res_x_err(2,:,1));
plot(time, res_x_err(2,:,2));
ylabel('p_y', 'fontsize', 12); grid on;
subplot(6,1,3); hold on;
plot(time, res_x_err(3,:,1));
plot(time, res_x_err(3,:,2));
ylabel('p_z', 'fontsize', 12); grid on;
subplot(6,1,4); hold on;
plot(time, res_x_err(4,:,1));
plot(time, res_x_err(4,:,2));
ylabel('v_x', 'fontsize', 12); grid on;
subplot(6,1,5); hold on;
plot(time, res_x_err(5,:,1));
plot(time, res_x_err(5,:,2));
ylabel('v_y', 'fontsize', 12); grid on;
subplot(6,1,6); hold on;
plot(time, res_x_err(6,:,1));
plot(time, res_x_err(6,:,2));
ylabel('v_z', 'fontsize', 12); xlabel('Time', 'fontsize', 12); grid on;






