clear all;close all;clc
img=im2double(imread('fish_in_the_stage13.png'));

img_ROI = img(150:350,380:580);

img_ROI(img_ROI>0.8)=0;

img_blur = imgaussfilt(img_ROI,4);

mask = im2double(img_blur<70/255);

imagesc(img_blur)