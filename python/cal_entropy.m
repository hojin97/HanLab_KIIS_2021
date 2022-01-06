clear all;
clc;
close all;

I = imread('./Images/GADF/Dumbbell_Curl/0_ang.png');
J = entropy(I(1))