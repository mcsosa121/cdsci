% Part 1. Data Generation
% Good link
% https://emtiyaz.github.io/pcml15/ex4-cv-and-bias-variance-decomposition.pdf
means = [[0;0],[0;1],[1;1],[1;0]];
sigmas = 0.3 * ones(1,4);
[Xtr,C] = MixGauss(means,sigmas,30);
figure;
% Scatter is scatter(locationX, locationY, circle sizes, colors)
%scatter(Xtr(:,1), Xtr(:,2),25,C);

% obtain 2 class training set with opposite corners sharing 
% the same class with labels +1 and -1
Ytr = 2*(1/2 - mod(C,2));

% This is called our training set [Xtr, Ytr]
% We now want to greate our test dataset
[Xte,C] = MixGauss(means,sigmas,200);
Yte = 2*(1/2 - mod(C,2));
% [Xte, Yte] is now our testing set.

% visualizing data
% scatter(Xtr(:,1), Xtr(:,2), 25, Ytr);
% scatter(Xte(:,1), Xte(:,2), 25, Yte);

% Part 2. KNN classification 

% WE can use knn to generate predictions for 2 class data
% knn(xtraining, ytraining, k (# of neighbors. usually odd), xtesting)
k = 3;
Ypr = kNNClassify(Xtr,Ytr,59,Xte);
% now we can estimate the training performance by comparing the predicted
% labels to the true labels
l = (Ypr ~= Yte);
err = sum(l)/length(Yte);
fprintf('Error with %d neighbors is: %d \n',k,err);

% visualizes the wrongly classified points
%separatingFkNN(Xtr,Ytr,k);
%hold on
%scatter(Xte(:,1), Xte(:,2), 25, Yte); % true label
%scatter(Xte(l,1), Xte(l,2), 25, Ypr(l),'r','x'); % color them
%hold off
% can use separating FkNN to visualize seperating function 
% or the areas of the 2D plane associated by the classifier
% with each class

% Up to now arbitrary choice for k. Can use holdoutCVkNN 
% for selection of model. 
% generate our k's
kv = 3:100;
kv = kv(rem(kv,2)==1);
[k, Vm, Vs, Tm, Ts] = holdoutCVkNN(Xtr, Ytr, 0.5, 15, kv);
errorbar(kv, Vm, sqrt(Vs)); hold on
errorbar(kv, Tm, sqrt(Ts)); axis tight;
