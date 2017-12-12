function separatingFKernRLS(c, Xtr, kernel, sigma, Xts, step)
    % function separatingFKernRLS(w, Xtr, kernel, sigma, Xts) 
    % 
    % Classifies points evenly sampled, according to a Kernel Regularized Least 
    % Squares classifier and plots the separating contour.
    %
    % c - coefficents of the function
    % Xtr - training examples
    % kernel, sigma - parameters used in learning the function
    % Xts - test examples on which to plot the separating function
    %
    % lambda = 0.01;
    % kernel = 'gaussian';
    % sigma = 1;
    % [Xtr, Ytr] = MixGauss([[0;0],[1;1]],[0.5,0.25],1000);
    % [Xts, Yts] = MixGauss([[0;0],[1;1]],[0.5,0.25],1000);
    % Ytr(Ytr==2) = -1;
    % Yts(Yts==2) = -1;
    % c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, lambda);
    % separatingFKernRLS(c, Xtr, kernel, sigma, Xts);

    if nargin<6, step = 0.05; end

    x = min(Xts(:,1)):step:max(Xts(:,1));
    y = min(Xts(:,2)):step:max(Xts(:,2));

    [X, Y] = meshgrid(x, y);
    XGrid = [X(:), Y(:)];

    YGrid = regularizedKernLSTest(c, Xtr, kernel, sigma, XGrid);

    hold on
    contour(x, y, reshape(YGrid,numel(y),numel(x)),[0;0]);
    hold off
end
