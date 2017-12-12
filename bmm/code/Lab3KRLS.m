% Lab 3
% Kernel Regulaized Least Squares
% Load data
A = load('data\moons_dataset.mat');
Xtr = A.Xtr;
Ytr = A.Ytr;
Xte = A.Xte;
Yte = A.Yte;

figure;
hold on;
scatter(Xtr(:,1),Xtr(:,2),25,Ytr);
scatter(Xte(:,1),Xte(:,2),25,Yte);
hold off;