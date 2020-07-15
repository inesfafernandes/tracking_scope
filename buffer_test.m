% Create a buffer of data to mimic an external data acquisition
x = buffer(1:1000, 50); % 50-sample nonoverlapping frames of data
 
% Loop over each frame of source data, to mimic the sequential
% arrival of each single frame of data:
 z = [];  opt = [];
 for i=1:size(x,2) % Loop over each source frame (column)
     acq = x(:,i);% Assume that this is what our data

                                   % acquisition board returns
    % y will contain a matrix of "rebuffered" data frames
    % NOTE: For the first loop iteration, z and opt are empty
    [y,z,opt] = buffer([z;acq], 24, 8, opt);
           % Do something with the buffer of data
    
 end
 
 figure(1)
 plot(y)
 figure(2)
 plot(x)
