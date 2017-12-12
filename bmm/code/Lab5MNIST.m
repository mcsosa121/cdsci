% Lab 5: Classifying MNIST data

% Loading the data and visualizing
A = load('data\MNIST_3_5.mat');
X = A.X; 
Y = A.Y;

% Show image
% visualizeExample(X(1,:));

% Split dataset into testing and training
[Xtr, Ytr, Xte, Yte] = randomSplitDataset(X,Y,100,1000);

% finding the optimal paramters
intLambda = (1:25)*0.01;
intKerPar = (1:25)*0.01;
[l, s, Vm, Vs, Tm, Ts] = ...
    holdoutCVKernRLS(Xtr, Ytr, 'gaussian', 0.5, 10, ...
                 intLambda, intKerPar);
%     X: N x D, the training examples
%     Y: N x 1, the training labels
%     kernel: string, the kernel function (see kernelMatrix.m).
%     perc: percentage of the dataset to be used for validation
%     nrip: number of repetitions of the validation for each parameter set
%     intLambda: list of regularization parameters
%     intKerPar: list of kernel parameters
%
% Output:
%     l, s: reg. and kernel parameter that minimize the mean (median ?) of the validation error
%     Vm, Vs: median and variance of validation error for each parameter pair
%     Tm, Ts: median and variance of training error for each parameter pair

% Train
C = regularizedKernLSTrain(Xtr, Ytr, 'gaussian', s, l);
Ykrls = regularizedKernLSTest(C, Xtr, 'gaussian', s, Xte);

ind = find((sign(Ykrls)~=sign(Yte)));
idx = ind(randi(numel(ind)));
figure; 
visualizeExample(Xte(idx,:));
