%% tracking objects

%% acquiring video

info= imaqhwinfo; % information about camera
%connecting video
vid = videoinput('winvideo', 1, 'RGB24_352x288');%(ADAPTORNAME,DEVICEID,FORMAT)

%how many frames we want to acquire and at what rate
frames=50;
frame_rate=3;
set(vid, 'FramesPerTrigger', frames);
set(getselectedsource(vid), 'FrameRate', frame_rate);

%starts acquiring images
start(vid)

%The processing does not need to start before the acquisition is done, so we’ll use the wait function to wait for the acquisition to stop
wait(vid)
[f, t] = getdata(vid);%transfers the acquired images into the MATLAB workspace.
% f represents the image sequence
% vector t contains the time stamps for each frame

%% frame differencing

%convert each frame to grayscale 
numframes = size(f, 4); %returns the lengths of the specified dimensions in a row vector (in this case dimension 4, which is frames)

%the backwards loop is a trick to ensure that g is initialized to its final
%size the first time through the loop, so that it doesnt keep changing size
%of the iterations

for k = numframes:-1:1
    g(:, :, k) = rgb2gray(f(:, :, :, k));
end

%background substraction

background = imdilate(g, ones(1, 1, 5));
%absolute difference between each frame and its corresponding background estimate
d = imabsdiff(g, background);
thresh = graythresh(d);
bw = (d >= thresh * 255); %Since graythresh returns a normalized value in the range [0,1], we must scale it to fit our data range, [0,255].

%compute the location of the ball in each frame

centroids = zeros(numframes, 2);
for k = 1:numframes
    L = bwlabel(bw(:, :, k));%we label each individual object (using bwlabel) and compute its corresponding center of mass (using regionprops).
    s = regionprops(L, 'area', 'centroid');%meausures a set of properties for each labeled region in the label matrix L
    area_vector = [s.Area];
    [tmp, idx] = max(area_vector);
    centroids(k, :) = s(idx(1)).Centroid;
end

%objects motion

subplot(2, 1, 1)
plot(t, centroids(:,1)), ylabel('x')
subplot(2, 1, 2)
plot(t, centroids(:, 2)), ylabel('y')
xlabel('time (s)')