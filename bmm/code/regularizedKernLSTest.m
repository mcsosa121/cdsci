function Yt = regularizedKernLSTest(C, X, kernel, param, Xt)
    % Test Kernel RLS
    %
    %  C: the (N x 1) vector of coefficients of the kernel representation (from
    %  training)
    %  X: the (N x D) training data matrix
    %  kernel: can be 'linear', 'polynomial' or 'gaussian'
    %  param: the kernel parameter

    %  Xt: the (M x D) test data matrix
    %  Yt: the (M x 1) output vector

    Ktest = KernelMatrix(Xt, X, kernel, param);

    Yt = Ktest*C; 
end
