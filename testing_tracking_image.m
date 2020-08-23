%% testing tracking sequence of images

foto1=rgb2gray(imread('foto1.jpg'));
foto2=rgb2gray(imread('foto2.jpg'));
foto3=rgb2gray(imread('foto3.jpg'));
foto4=rgb2gray(imread('foto4.jpg'));
foto5=rgb2gray(imread('foto5.jpg'));

figure(1)
subplot(1,5,1)
imshow(foto1)
subplot(1,5,2)
imshow(foto2)
subplot(1,5,3)
imshow(foto3)
subplot(1,5,4)
imshow(foto4)
subplot(1,5,5)
imshow(foto5)

background1 = imdilate(foto1, ones(20, 20, 5));
background2 = imdilate(foto2, ones(20, 20, 5));
background3 = imdilate(foto3, ones(20, 20, 5));
background4 = imdilate(foto4, ones(20, 20, 5));
background5 = imdilate(foto5, ones(20, 20, 5));
%absolute difference between each frame and its corresponding background estimate
d1 = imabsdiff(foto1, background1);
d2 = imabsdiff(foto2, background2);
d3 = imabsdiff(foto3, background3);
d4 = imabsdiff(foto4, background4);
d5 = imabsdiff(foto5, background5);

thresh1 = graythresh(d1);
thresh2 = graythresh(d2);
thresh3 = graythresh(d3);
thresh4 = graythresh(d4);
thresh5 = graythresh(d5);

bw1 = (d1 >= thresh1 * 255);
bw2 = (d2 >= thresh2 * 255);
bw3 = (d3 >= thresh3 * 255);
bw4 = (d4 >= thresh4 * 255);
bw5 = (d5 >= thresh5 * 255);

L1 = bwlabel(bw1);
L2 = bwlabel(bw2);
L3 = bwlabel(bw3);
L4 = bwlabel(bw4);
L5 = bwlabel(bw5);
s1 = regionprops(L1, 'area', 'centroid');
s2 = regionprops(L2, 'area', 'centroid');
s3 = regionprops(L3, 'area', 'centroid');
s4 = regionprops(L4, 'area', 'centroid');
s5 = regionprops(L5, 'area', 'centroid');
area_vector1 = [s1.Area];
area_vector2 = [s2.Area];
area_vector3 = [s3.Area];
area_vector4 = [s4.Area];
area_vector5 = [s5.Area];
[tmp1, idx1] = max(area_vector1);
[tmp2, idx2] = max(area_vector2);
[tmp3, idx3] = max(area_vector3);
[tmp4, idx4] = max(area_vector4);
[tmp5, idx5] = max(area_vector5);
centroids(1, :) = s1(idx1(1)).Centroid;
centroids(2, :) = s2(idx2(1)).Centroid;
centroids(3, :) = s3(idx3(1)).Centroid;
centroids(4, :) = s4(idx4(1)).Centroid;
centroids(5, :) = s5(idx5(1)).Centroid;

t=[1:5];

%axis origin is on top left corner
%  --------->x
%  |
%  |
%  |
%  |
%  ^y
[lin,col]=size(foto1);
y_real= lin-centroids(:,2);

figure(2)
subplot(2, 1, 1)
plot(t, centroids(:,1)), ylabel('x')
subplot(2, 1, 2)
plot(t, y_real), ylabel('y')
xlabel('time (s)')

figure(3)

plot(centroids(:,1),y_real)