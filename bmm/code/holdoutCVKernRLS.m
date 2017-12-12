function [l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(X, Y, kernel, perc, nrip, intLambda, intKerPar, verbose)
    % Hold-out Cross-validation for Kernel RLS
    %
    % [l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(X, Y, kernel, perc, nrip, intLambda, intKerPar)
    %
    % Input:
    %     X: N x D, the training examples
    %     Y: N x 1, the training labels
    %     kernel: string, the kernel function (see kernelMatrix.m).
    %     perc: percentage of the dataset to be used for validation
    %     nrip: number of repetitions of the validation for each parameter set
    %     intLambda: list of regularization parameters
    %     intKerPar: list of kernel parameters
    %
    % Output:
    %     l, s: reg. and kernel parameter that minimize the mean (median ?) of the validation error
    %     Vm, Vs: median and variance of validation error for each parameter pair
    %     Tm, Ts: median and variance of training error for each parameter pair
    %
    % Example:
    % intLambda = [5,2,1,0.7,0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00001,0.000001];
    % intKerPar = [10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1,0.05,0.03,0.02,0.01];
    % [Xtr, Ytr] = MixGauss([[0;0],[1;1]],[0.5,0.25],100);
    % [l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(Xtr, Ytr,'gaussian', 0.5, 5, intLambda, intKerPar);
    % plot(intLambda, Vm, intLambda, Tm);

    if nargin<8, verbose = true; end

    nKerPar = numel(intKerPar);
    nLambda = numel(intLambda);

    n = size(X,1);
    ntr = ceil(n*(1-perc));

    ym = (max(Y) + min(Y))/2;

    trerr = zeros(nLambda, nKerPar, nrip);
    vlerr = zeros(nLambda, nKerPar, nrip);

    for rip = 1:nrip

        % Random splits
        I = randperm(n);

        % Training set
        Xtr = X(I(1:ntr), :);
        Ytr = Y(I(1:ntr), :);
        % Validation set
        Xvl = X(I(ntr + 1:end), :);
        Yvl = Y(I(ntr + 1:end), :);

        il = 0;
        for l = intLambda
            il = il + 1;
            is = 0;
            for s = intKerPar
                is = is + 1;

                % Run Kernel RLS with (l, s) parameter pair
                c = regularizedKernLSTrain(Xtr, Ytr, kernel, s, l);

                trerr(il, is, rip) = calcErr(regularizedKernLSTest(c, Xtr, kernel, s, Xtr), Ytr, ym);
                vlerr(il, is, rip) = calcErr(regularizedKernLSTest(c, Xtr, kernel, s, Xvl), Yvl, ym);

                if verbose
                    fprintf('iter: %3d, lambda: %02.5f, sigma: %0.3f, valErr: %0.3f, trErr: %0.3f\n', rip, l, s, vlerr(il, is, rip), trerr(il, is, rip));
                end
            end
        end

    end

    % Mean and std over random splits/multiple trials
    Tm = mean(trerr, 3);
    Ts = std(trerr, [], 3);
    Vm = mean(vlerr, 3);
    Vs = std(vlerr, [], 3);

    %[row, col] = find(Vm + sqrt(Vs) <= min(min(Vm+sqrt(Vs))));
    %l = intLambda(row(1));
    %s = intKerPar(col(1));

    % Choose minimum mean validation 
    [row, col] = find(Vm == min(min(Vm)));
    % if multiple (global) minimizers choose the 'largest' lambda, i.e. 'smoothest'/simplest model 
    [l, m] = max(intLambda(row));
    % sigma for chosen lambda
    s = intKerPar(col(m));
end

function err = calcErr(T, Y, m)
    vT = (T >= m);
    vY = (Y >= m);
    err = sum(vT ~= vY)/numel(Y);
end
