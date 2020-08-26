%% Kalman filter
%https://www.datasciencecentral.com/profiles/blogs/using-bayesian-kalman-filter-to-predict-positions-of-moving
% tutorial (https://www.youtube.com/watch?v=9Ilb4_DJB_s)

%% representation  of the state-space model
%dx/dt=Ax(t)+Bu(t)
%y= Cx(t)
A=
B=
C=
sys= ss(A,B,C);

%% building the kalman filter

%covariance matrixes
%file:///C:/Users/TeachinglabA/Downloads/Discrete-timeKalmanfilter.pdf
%qn=E(wk.wk'), let us assume [1 0 ; 0 1], wk is the system noise vector
%rn=E(vk^2), let us assume 1, vk is the  measurement noise vecto
qn=[1 0; 0 1];
rn=1;
%KEST generates optimal estimates y_e(t) and x_e(t)
%L is the estimator gain
%P is the steady state error covariance
[KEST,L,P]=kalman(sys,qn,rn);
