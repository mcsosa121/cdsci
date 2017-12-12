function K = KernelMatrix(X1, X2, kernel, param)
    %
    % K = KernelMatrix(X1, X2, kernel, param) computes the N x M kernel matrix
    % from matrices X1 (size N x D) and X2 (size M x D).

    % kernel: string, can be 'linear', 'polynomial' or 'gaussian'
    % param: is [] for the linear, the exponent of the polynomial, or the
    % variance for the gaussian kernel.

    if isempty(kernel), kernel = 'linear'; end

    switch kernel
        case 'linear'
            K = X1*X2';
        case 'polynomial'
            K = (1 + X1*X2').^param;
        case 'gaussian'
            K = exp(-1/(2*param^2)*SquareDist(X1, X2));
        otherwise
            error('Unrecognized kernel type. \"kernel\" can be \"linear\", \"polynomial\" or \"gaussian\"');
    end
end
