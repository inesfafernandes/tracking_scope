%% testing tracking sequence of images

foto1=rgb2gray(imread('foto1.jpg'));
foto2=rgb2gray(imread('foto2.jpg'));
foto3=rgb2gray(imread('foto3.jpg'));

figure(1)
subplot(1,3,1)
imshow(foto1)
subplot(1,3,2)
imshow(foto2)
subplot(1,3,3)
imshow(foto3)

background1 = imdilate(foto1, ones(20, 20, 5));
background2 = imdilate(foto2, ones(20, 20, 5));
background3 = imdilate(foto3, ones(20, 20, 5));
%absolute difference between each frame and its corresponding background estimate
d1 = imabsdiff(foto1, background1);
d2 = imabsdiff(foto2, background2);
d3 = imabsdiff(foto3, background3);

thresh1 = graythresh(d1);
thresh2 = graythresh(d2);
thresh3 = graythresh(d3);

bw1 = (d1 >= thresh1 * 255);
bw2 = (d2 >= thresh2 * 255);
bw3 = (d3 >= thresh3 * 255);

L1 = bwlabel(bw1);
L2 = bwlabel(bw2);
L3 = bwlabel(bw3);
s1 = regionprops(L1, 'area', 'centroid');
s2 = regionprops(L2, 'area', 'centroid');
s3 = regionprops(L3, 'area', 'centroid');
area_vector1 = [s1.Area];
area_vector2 = [s2.Area];
area_vector3 = [s3.Area];
[tmp1, idx1] = max(area_vector1);
[tmp2, idx2] = max(area_vector2);
[tmp3, idx3] = max(area_vector3);
centroids(1, :) = s1(idx1(1)).Centroid;
centroids(2, :) = s2(idx2(1)).Centroid;
centroids(3, :) = s3(idx3(1)).Centroid;

t=[1:3];

figure(2)
subplot(2, 1, 1)
plot(t, centroids(:,1)), ylabel('x')
subplot(2, 1, 2)
plot(t, centroids(:, 2)), ylabel('y')
xlabel('time (s)')

figure(3)

plot(centroids(:,1),centroids(:,2))