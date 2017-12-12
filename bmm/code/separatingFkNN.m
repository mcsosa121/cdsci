function separatingFkNN(Xtr, Ytr, k, step)
    % function separatingF(Xtr,Ytr,k)
    %
    % Classifies points on a regular grid sampled using k-Nearest Neighbors
    % (kNNClassify) and plots the separating contour.
    %
    % Xtr - training examples
    % Ytr - training labels
    % k - number of neighbors for kNN
    % step - spacing in x,y directions for grid: 0.1 (default)
    %
    %
    % Example:
    % [Xtr, Ytr] = MixGauss([[0;0],[1;1]],[0.5,0.25],100); Ytr = mod(Ytr,2)*2-1;
    % [Xts, Yts] = MixGauss([[0;0],[1;1]],[0.5,0.25],100); Yts = mod(Yts,2)*2-1;
    % figure;
    % k = 5;
    % separatingFkNN(Xtr, Ytr, k); hold on
    % scatter(Xts(:, 1), Xts(:, 2), 25, Yts, 'filled');

    if nargin<4, step = 0.1; end

    % Define regular grid using training set X
    x = min(Xtr(:,1)):step:max(Xtr(:,1));
    y = min(Xtr(:,2)):step:max(Xtr(:,2));

    [X, Y] = meshgrid(x, y);
    XGrid = [X(:), Y(:)];

    % Classify points in the grid
    YGrid = kNNClassify(Xtr, Ytr,  k, XGrid);

    % Draw contour
    lw = 1; c = 'k';
    contour(x, y, reshape(YGrid, numel(y), numel(x)), [0,0], 'linewidth', lw, 'Color', c);
    axis image;
end
