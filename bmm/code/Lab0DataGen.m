% helpful link 
% https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
means = [[0;0], [1;1]];
sigmas = [0.5, 0.25];

% visualize simple dataset
[X,C] = MixGauss(means,sigmas, 1000);
%figure;
%scatter(X(:,1), X(:,2),25,C);

% More complex datasets
% 4 class dataset: Classes must live in 2D space and be centered
% on the corners of the unit sq (0,0), (0,1), (1,1), (1,0) all with
% variance 0.2.
means4 = [[0;0],[0;1],[1;1],[1;0]];
sigmas4 = [0.2,0.2,0.2,0.2];
[X4,C4] = MixGauss(means4,sigmas4,1000);
%figure;
%scatter(X4(:,1), X4(:,2),25,C4);

% Now manipulate the data to obtain a 2-class problem where data
% on opposite corners share the same class
% Can manipulate C4 to get 2 classes. 
Y = mod(C4,2);
figure;
scatter(X4(:,1), X4(:,2),25,Y);


