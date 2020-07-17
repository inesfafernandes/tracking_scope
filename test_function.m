N=1e5;%tem de ser igual ao dq.rate?
F=1000;
K=100;

h=exp(-(1:K)/10);
x=randn(N,1);%input lido
y=conv(x,h,'valid')+rand(N-K+1,1);%output lido

H=system_identification_fc(x,y);

figure(1)
plot(h-H')

figure(2)
plot(h); title(' True H vs estimate H')
hold on
plot(H)
legend('True H', 'estimated H');