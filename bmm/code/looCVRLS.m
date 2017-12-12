function [lambda, Vm, Tm] = looCVRLS(X, Y, intLambda, verbose)
    % Leave-One-Out Cross-validation for RLS
    %
    % Input:
    %     X: dataset (test set excluded)
    %     Y: labels (test set excluded)
    %     intLambda: vector of regularization parameters to search over
    %     verbose: true (default) for printing out information
    %
    % Output:
    %     lambda: value lambda in intLambda that minimizes the mean of the validation error
    %     Vm: validation error
    %     Tm: mean of the error computed on the training set
    %
    % Example:
    % intLambda = kron(10.^(-1:-1:-3), (5:-2:1));
    % [X, Y] = MixGauss([[0;0],[1;1]],[0.5,0.25],1000);
    % Y(Y==2)=-1;
    % [lambda, Vm, Tm] = looCVRLS(X, Y, intLambda);
    % plot(intLambda, Vm,, intLambda, Tm);

    if nargin<4, verbose = true; end

    n = size(X, 1);
    nLambda = numel(intLambda);
    Tm = zeros(1, nLambda);
    Vm = zeros(1, nLambda);

    ym = (max(Y) + min(Y))/2;

    for ir = 1:n

        % Training set
        Xtr = [X(1:ir-1, :); X(ir+1:end, :)];
        Ytr = [Y(1:ir-1, :); Y(ir+1:end, :)];

        % Validation set
        Xvl = X(ir, :);
        Yvl = Y(ir, :);

        for il = 1:nLambda

            lambda = intLambda(il);
            w = regularizedLSTrain(Xtr, Ytr, lambda);

            Tm(il, ir) = calcErr(regularizedLSTest(w, Xtr), Ytr, ym);
            Vm(il, ir) = calcErr(regularizedLSTest(w, Xvl), Yvl, ym);

        end
        if verbose
            fprintf('lambda: %0.5f, valErr: %0.3f, trErr: %0.3f\n', lambda, sum(Vm(il, :))/n, sum(Tm(il, :))/n);
        end
    end

    % Mean over trials
    Tm = sum(Tm,2)/n;
    Vm = sum(Vm,2)/n;

    % Optimum lambda selection (choose max lambda, i.e. smoother model)
    lambda = max(intLambda(Vm == min(Vm(:))));
end

function err = calcErr(T, Y, m)
    vT = (T >= m);
    vY = (Y >= m);
    err = sum(vT ~= vY)/numel(Y);
    % err = norm(vT - vY)/numel(Y);
end
