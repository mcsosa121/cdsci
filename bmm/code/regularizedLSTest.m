function Yt = regularizedLSTest(w, Xt)
    % Test RLS 
    %
    %  w: the (D x 1) vector of RLS coefficients (outut of regularizedLSTrain)
    %  Xt: the (M x D) test data matrix
    %  Yt: the (M x 1) output vector
    Yt = Xt*w;
end