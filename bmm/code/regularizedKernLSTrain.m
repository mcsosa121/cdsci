function C = regularizedKernLSTrain(X, Y, kernel, param, lambda)
    % Train Kernel RLS 
    % 
    %  X: the (N x D) training data matrix
    %  Y: the (N x 1) training output/label vector
    %  kernel: can be 'linear', 'polynomial' or 'gaussian'
    %  param: the kernel parameter 
    %  lambda: regularization parameter
    %
    %  C: the (N x 1) vector of coefficients of the kernel representation

    n = size(X, 1);
    K = KernelMatrix(X, X, kernel, param);

    C = (K+eye(n)*n*lambda) \ Y;
end
