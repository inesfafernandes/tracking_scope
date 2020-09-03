
function acc= accuracy_calculator(RMSE_model,thres)

acc=0;
for i=1:length(RMSE_model)
    if RMSE_model(i)<=thres
        acc=acc+1;
    end
end

