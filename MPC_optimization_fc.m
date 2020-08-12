function u = MPC_optimization_fc (fish_trajectory,h,T)
    
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
    %xstar=fish_trajectory;%sin(linspace(0,6*pi,T)');%sawtooth(linspace(0,8*pi,T)');%1; % position we want to reach 
    up=zeros(N-1,1); %u past
    I=eye(T); %identity matrix with size 20x20
    uf=zeros(1,T); %initializing u future matrix
    u=up;
    model=ones(T,1);

    %building matrix phi, phi_a and phi_b

    for i=2:T
        for j=2:M
            phi(i,j)=phi(i-1,j-1);
        end
    end

    phi_a(2:T,:)=phi(2:T,1:N-1);% sliding window of h for past values
    phi_b(2:T,:)=phi(2:T,N:end); %sliding window of h for future values
    
    const_up=((lambda*I+phi_b'*phi_b)\phi_b');
   
    for t=1:length(fish_trajectory)-T
        predicted_fish_trajectory=fish_trajectory(t:t+T-1);
        predicted_stage_trajectory=phi_a*up;
        predicted_error=predicted_fish_trajectory-predicted_stage_trajectory;
        uf=const_up*(predicted_error);%computing u future x axis
        %uf_y=const_up*((fish_trajectory_y(t)*model)-phi_a*up); %computing u future y axis
        up=cat(1,up,uf(1));% updating u past by adding the first element of uf as the last element of u past
        up(1)=[];% and discarding the first value of u past
        u=cat(1,u,uf(1));%vector that contains all commands that were sent
    end

end