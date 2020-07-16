clear all;close all;clc
N=1e5;
F=1000;
K=100;

h=exp(-(1:K)/10);
x=randn(N,1);%input lido
y=conv(x,h,'valid')+rand(N-K+1,1);%output lido

X=zeros(N-K+1, K); %matriz para calculo de H
%Y=y(K:end)
Y=y(1:end); %vetor para 
%Y=zeros(N-K+1,1);
%H=zeros(N-K+1,1);

for i=0:(N-K)
   X(i+1,:)=x(K+i:-1:K-(K-i)+1);
end

H=X\Y;
figure(1)
plot(h-H')

figure(2)
plot(h)
hold on
plot(H)