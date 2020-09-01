function [] = export_mpc_parms ()
T = 250;
N = size(h,1);  %past time
M = T+N-1; % total window of time

lambda = 1e-4; %constant value to minimize variation of voltage

phi = zeros(T,M);


phi(1,1 : N) = h(end:-1:1);


% replace with hankel function
for i=2:T
    for j=2:M
        phi(i,j)=phi(i-1,j-1);
    end
end


% sliding window of h for past values
phi_past = phi(:,1 : N-1);

%sliding window of h for future values
phi_future = phi(:,N : end);


response_constant = ((lambda*eye(T)+phi_future'*phi_future)\phi_future');

save('mpc_params.mat','phi_past','phi_future','response_constant','T','N','M');

