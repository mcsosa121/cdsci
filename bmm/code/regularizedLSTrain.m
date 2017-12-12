function w = regularizedLSTrain(X, Y, lambda)
    % RLS train
    %
    %  X: the (N x D) training data matrix
    %  Y: the (N x 1) training output/label vector
    %  lambda: regularization parameter
    %
    %  w: the (D x 1) vector of RLS coefficients
    % least sq is min 1/n * sum (yi - w'*xi)^2 + lambda*w'*w, 
    % with lambda >= 0 
    % Problem formatted as
    % (X'*X + lambda*n*I)*w = X'*Y
    [N, D] = size(X);

    % Solution for this problem can be found to be
    % w = X'*(X*X' + lambda*n*I)^-1 * Y 
    % naive way
    % xx = X*X';
    % xx = xx + lambda*N*eye(N);
    % xiy = xx \ Y;
    % w = X'*xiy;

    % SVD way
    % useful link
    % http://www.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_10_Linear_least_squares_reg.pdf
    [U,S,V] = svd(X);
    sig = diag(S).^2 + lambda*N;
    sig = sig.^-1;
    vs = V*diag(sig);
    vs = vs*S'*U';
    w= vs*Y;
end