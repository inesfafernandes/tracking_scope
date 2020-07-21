%% system identification function
%calculates an estimate of the stage kernel (H)

function H = system_identification_fc (x_input,y_output,K)
    N=length(x_input);
    
    x=x_input;
    y=y_output;
    
    X=zeros(N-K+1, K); %matriz para calculo de H
    Y=y(K:end);
    
    for i=0:(N-K)
        X(i+1,:)=x(K+i:-1:K-(K-i)+1);
    end

    H=X\Y;
    
    figure;plot(H);title('H');
    %saveas(gcf,'stage response_K_rate.fig');
end

