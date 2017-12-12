function [X, Y] = AnisotropicMixGauss(means, covs, n)
    % Generate a mixture of p anisotropic Gaussians with generic covariances
    %
    % means: (size d x p) stores the p mean vectors, i.e. is of the form [m1, ..., mp]
    % with means(:, i) = mi  (each mi is d-dimensional)
    % covs: (size d x d x p) stores the p covariance matrices, i.e. is of the
    % form covs(:, :, i) = sigma_i.
    % n: number of points per class
    %
    % X: data matrix (size 2n x d)
    % Y: label vector (size 2n) in [1, p]
    %
    % see also MixGauss.m

    nComp = size(means, 2); % number of components in mixture

    X = []; Y = [];
    for c=1:nComp
        [Xc, Yc] = AnisotropicGauss(means(:,c), covs(:,:,c), c, n);
        X = [X; Xc];
        Y = [Y; Yc];
    end
end

function [X, Y] = AnisotropicGauss(meanVec, covMat, labelClass, n)
    % Generate Anisotropic Gaussian

    d = numel(meanVec);
    X = randn(n, d)*covMat' + ones(n, 1)*meanVec';
    Y = ones(n,1)*labelClass;
end