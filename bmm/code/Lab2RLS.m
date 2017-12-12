% Regularized Least Squares

% 1. Classification Data Generation
means = [[-0.5;-0.5],[0.5;0.5]];
sigmas = [0.5,0.5];
% Generating training data

[Xtr,Ytr] = MixGauss(means,sigmas,5);
% Adjusting the output labels to be {1,-1}
Ytr(Ytr==2)=-1;

% Test Data
[Xte, Yte] = MixGauss(means,sigmas,200);
% Adding noise to the test data
Ytr = flipLabels(Ytr,0.2);
Yte = flipLabels(Yte,0.2);

figure;
%scatter(Xtr(:,1),Xtr(:,2),25,Ytr);
%hold on;
%scatter(Xte(:,1),Xte(:,2),25,Yte);
%hold off;

% 2. RLS Classification
w = regularizedLSTrain(Xtr,Ytr,0.1);
Yrls = regularizedLSTest(Xte,w);
err = sqrt((Yrls-Yte).^2);
z = zeros(length(Yte),1);
z(err <= 0.5)=1;
scatter(Xte(:,1),Xte(:,2),25,Yte);
hold on;
scatter(Xte(:,1),Xte(:,2),25,z,'x');
hold off;
