clear all
close all
%demo of motion predictive control approach copied from Drew and Jen's
%Julia version to Matlab 
% December 5th 2019 Michael Orger adapted from Drew Robson and Jennifer Li

% Here I load the stage velocity impulse response. This is a prediction of
% the resulting velocity change if you send a velocity command of 1 unit
% for 1 timestep. This is measured by sending a white noise command and
% correlating this input with the output. This gives you the optimal linear
% estimate of the stage impulse response ("Wiener-Volterra method").
load('drewjenimpulsedata.mat')

%simulated trajectory of a fish for 75 timesteps. This guy jumps from
%position 0 to position 1 at frame 26
fish_position=[zeros(25,1); ones(50,1)];

%or try this random movement example
% fish_position=randn(4750,1);
% fish_position=conv(fish_position,ones(100,1),'same');

%Here we simulate a 2 frame delay in the camera. This is hacked in
%in the Julia example, in a way that wouldn't let you experiment with other
%examples
recorded_fish_position=fish_position;
recorded_fish_position(3:end)=recorded_fish_position(1:end-2);

%This will record the actual stage_position through the experiment
stage_position=zeros(size(fish_position));

%This will record the velocity command sent to the stage
control_input=zeros(size(fish_position));

%Here we set the control and prediction horizons. This method estimates how
%it is going to control the stage for n time steps ahead, based on a
%prediction of the required trajectory, allowing anticipation of movement
%and compensation for delays in the system. The 'control horizon'
%determines how many control points ahead you will try to plan. The
%'prediction horizon' is how many timesteps ahead you try to predict the motion to
%evaluate the control sequence.
ctrl_hrz=4;
pred_hrz=7;

% This matrix sets a regularization parameter that aims to avoid extreme movements
% of the stage (I will improve the explanation here when I have gone
% through it myself)
% you might need to play with this value to avoid runaway control signals
mu=1e-6;
%gg is a square matrix with value 'mu' along the diagonal and 0 everywhere else/
gg=mu*eye(ctrl_hrz);

%A is a matrix that acts on a velocity control sequence (length ctrl_hrz) to generate a predicted
%sequence of position changes length (length pred_hrz)
A=zeros(pred_hrz,ctrl_hrz);
for i=1:ctrl_hrz
    % Each column of A is the cumulative sum of the velocity impulse
    % response - this gives you the expected POSITION that results from
    % that velocity impulse. Each successive column is shifted down one step because
    % each velocity command arrives one timestep later.
   A(i:end,i)=cumsum(imp_resp_x(1:pred_hrz-i+1));
end

%As above, the cumulative sum of teh velocity impulse gives an expected
%position step.
stpResp=cumsum(imp_resp_x);

% now we iterate through 50 frames of simulated tracking
for i=1:length(fish_position)-25
    
    %gets the current estimate of the future stage position
   predicted_stage_position=stage_position(i+[1:pred_hrz]);
   
   % gets the desired position ie the position of the fish, but here
   % recorded with the camera delay
   target_stage_position=recorded_fish_position(i+[1:pred_hrz]);
   %no knowledge of the future!
   target_stage_position(2:end)=target_stage_position(1);
   
   %optional: try extrapolating the fish trajectory a little
%    if i>1
%        fishVelocity=(recorded_fish_position(i+1)-recorded_fish_position(i-1))/2;
%        target_stage_position(2)=target_stage_position(2)+fishVelocity;
%        target_stage_position(3:end)=target_stage_position(3:end)+2*fishVelocity;
%    end
   
   % calculate the prediction error based on our current control estimate
   predicted_error = target_stage_position - predicted_stage_position;
   
   %calculate up to ctr_hrz the optimal control signal to reduce this error
   optimal_control = (A'*A + gg) (A'*predicted_error);
   
   % we will only apply the control for the very next frame
   control_input(i) = optimal_control(1);
   
   % update the predicted stage position based on the new control time
   % step. ie add the scaled position step response to the upcoming frames.
   for j = i + 1 : length(stage_position)
        if (j - i > length(imp_resp_x))  break; end
    stage_position(j) = stage_position(j)+control_input(i) * stpResp(j - i);
   end
   
   
end

figure 
plot(fish_position(1:length(fish_position)-25),'r')
hold on
plot(stage_position(1:length(fish_position)-25),'g')
plot(control_input(1:length(fish_position)-25)/1000,'b')
legend('Fish Position','Stage Position','Control Signal')
trackingerror=abs(fish_position(1:length(fish_position)-25)-stage_position(1:length(fish_position)-25));
title(strcat('Median tracking error:  ',num2str(median(trackingerror))));