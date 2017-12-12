function s = autosigma(X, K)
    %AUTOSIGMA Compute the average K nearest neighbor distance of n p-dimensional points.
    %
    %   S = AUTOSIGMA(X, K) calculates the average K nearest neighbor
    %   distance in p-dimensions given a data matrix 'X[n,p]' and a number
    %   'K' of nearest neighbors
    %
    %   Example:
    %        s = autosigma(X, 5);
    %
    % See also KERNELMATRIX

    % Euclidean distance
    E = real(sqrt(SquareDist(X, X)));
    % Sort, i.e. nearest neighbors/smallest distance per point
    E = sort(E);
    % Average of the K nearest neighbors
    s = mean(mean(E(2:K+1, :)));
end
