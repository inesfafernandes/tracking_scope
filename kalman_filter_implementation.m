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
%x(t+1)=x+u*dt+n, something that will move acording to velocity
%y= x+v ,measured position

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

%other optio to check



