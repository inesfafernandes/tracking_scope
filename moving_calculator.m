%%moving calculator

function [under_100_not_moving,under_100_moving,velocity_filt]= moving_calculator(x_trajectory, output_trajectory)

threshold_vel=0.0005;%value below which the fish is not considered to be  moving
threshold_error=0.01;% cm
not_moving=[];
moving=[];

if length(x_trajectory)<length(output_trajectory)
    N=abs(length(x_trajectory)-length(output_trajectory));
    output_trajectory=output_trajectory(N+1:end);
    velocity_filt= diff(sgolayfilt(output_trajectory,2,13));
    error_direct_command=x_trajectory-output_trajectory;
    
    % divide into 2 groups moving and not moving based on a velocity
    % threshold
    for i=1:length(velocity_filt)
        if velocity_filt(i)<threshold_vel
            not_moving=[not_moving i];
        else
            moving=[moving i];
        end
    end

    under_100_not_moving=0;
    under_100_moving=0;
    
    %in the not moving group divide between the ones that have error under
    %and over 100micrometers
    for i=not_moving
        if error_direct_command(i)<=threshold_error
            under_100_not_moving=under_100_not_moving+1;
        end
    end
    
    %in the moving group divide between the ones that have error under
    %and over 100micrometers
    for i=moving
        if error_direct_command(i)<=threshold_error
            under_100_moving=under_100_moving+1;
        end
    end
else
    M=abs(length(x_trajectory)-length(output_trajectory));
    x_trajectory=x_trajectory(1:end-M);
    velocity_filt= diff(sgolayfilt(output_trajectory,2,13));
    error_direct_command=x_trajectory-output_trajectory;
    
    % divide into 2 groups moving and not moving based on a velocity
    % threshold
    for i=1:length(velocity_filt)
        if velocity_filt(i)<threshold_vel
            not_moving=[not_moving i];
        else
            moving=[moving i];
        end
    end

    under_100_not_moving=0;
    under_100_moving=0;
    
    %in the not moving group divide between the ones that have error under
    %and over 100micrometers
    for i=not_moving
        if error_direct_command(i)<=threshold_error
            under_100_not_moving=under_100_not_moving+1;
        end
    end
    
    %in the moving group divide between the ones that have error under
    %and over 100micrometers
    for i=moving
        if error_direct_command(i)<=threshold_error
            under_100_moving=under_100_moving+1;
        end
    end
end