%%error

function [error_count,max_error,min_error]= error_calculator(error)

max_error=max(abs(error));
min_error=min(abs(error));

count_1_3=0;
count_2_3=0;
count_3_3=0;

for i=1:length(error)
    if abs(error(i))<= (max_error/3)
        count_1_3=count_1_3+1;
    elseif (max_error/3)<abs(error(i)) && abs(error(i))<=(2/3*(max_error))
        count_2_3=count_2_3+1;
    else
        count_3_3=count_3_3+1;
    end    
end
        
error_count=[(count_1_3/length(error))*100 (count_2_3/length(error))*100 (count_3_3/length(error))*100] ;
end
