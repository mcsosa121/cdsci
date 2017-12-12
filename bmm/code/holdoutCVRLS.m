function [lambda, Vm, Vs, Tm, Ts] = holdoutCVRLS(X, Y, perc, nrip, intLambda, verbose)
    % Hold-Out Cross-validation for RLS
    %
    % Input:
    %     X: dataset (test set excluded)
    %     Y: labels (test set excluded)
    %     perc: percentage of the dataset to be used for validation
    %     nrip: number of repetitions of the validation
    %     intLambda: vector of regularization parameters to search over
    %     verbose: true (default) for printing out information
    %
    % Output:
    %     lambda: the value lambda in intLambda that minimize the mean of the
    %           validation error
    %     Vm, Vs: mean and variance of the validation error for each param
    %     Tm, Ts: mean and variance of the error computed on the training set for
    %     each param.
    %
    % Example:
    % intLambda = kron(10.^(-1:-1:-3), (5:-2:1));
    % [X, Y] = MixGauss([[0;0],[1;1]],[0.5,0.25],1000);
    % Y(Y==2)=-1;
    % [lambda, Vm, Vs, Tm, Ts] = holdoutCVRLS(X, Y, 0.5, 10, intLambda);
    % plot(intLambda, Vm, intLambda, Tm);

    if nargin<6, verbose = true; end

    nLambda = numel(intLambda);

    n = size(X,1);
    ntr = ceil(n*(1-perc));

    ym = (max(Y) + min(Y))/2;

    trerr = zeros(nLambda, nrip);
    vlerr = zeros(nLambda, nrip);

    for ir = 1:nrip

        % Random splits
        I = randperm(n);

        % Training set
        Xtr = X(I(1:ntr), :);
        Ytr = Y(I(1:ntr), :);
        % Validation set
        Xvl = X(I(ntr + 1:end), :);
        Yvl = Y(I(ntr + 1:end), :);

        for il = 1:nLambda

            lambda = intLambda(il);
            w = regularizedLSTrain(Xtr, Ytr, lambda);

            trerr(il, ir) = calcErr(regularizedLSTest(w, Xtr), Ytr, ym);
            vlerr(il, ir) = calcErr(regularizedLSTest(w, Xvl), Yvl, ym);

            if verbose
                fprintf('iter: %3d, lambda: %02.4f, valErr: %0.3f, trErr: %0.3f\n', ir, lambda, vlerr(il, ir), trerr(il, ir));
            end
        end
    end

    % Mean and std over random splits/multiple trials
    Tm = mean(trerr, 2);
    Ts = std(trerr, [], 2);
    Vm = mean(vlerr, 2);
    Vs = std(vlerr, [], 2);

    % Optimum lambda selection (choose max lambda, i.e. smoother model)
    lambda = max(intLambda(Vm == min(Vm(:))));
end

function err = calcErr(T, Y, m)
    vT = (T >= m);
    vY = (Y >= m);
    err = sum(vT ~= vY)/numel(Y);
    % err = norm(vT - vY)/numel(Y);
end
