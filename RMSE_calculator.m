%%RMSE
function RMSE= RMSE_calculator(xstar,model)
RMSE=[];

if length(xstar)<length(model)
    N=abs(length(xstar)-length(model));
    model=model(N+1:end);
    for i=1:length(model)
        rmse=sqrt(mean((xstar(i)-model(i))^2));
        RMSE=cat(1,RMSE,rmse);
    end
else
    M=length(xstar)-length(model);
    xstar=xstar(1:end-M);
    for i=1:length(xstar)
        rmse=sqrt(mean((xstar(i)-model(i))^2));
        RMSE=cat(1,RMSE,rmse);
    end
    
end

