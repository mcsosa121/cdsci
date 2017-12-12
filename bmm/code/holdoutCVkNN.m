function [k, Vm, Vs, Tm, Ts] = holdoutCVkNN(X, Y, perc, nrip, intK)
    % Hold-out Cross-validation for kNN classifier
    %
    % X: train dataset
    % Y: train labels 
    % perc: percentage of data to be used for validation
    % nrip: number of repetitions of validation (random splits)
    % intK: vector of K (nearest neighbors) values to search over, e.g. intK = [1:2:11 17 21:10:51]
    %
    % Output:
    % k: value k in intK that minimizes the mean of the validation error
    % Vm, Vs: mean and variance of error on validation set  
    % Tm, Ts: mean and variance of error on training set  
    %
    % Example: 
    % intK = [1:2:11 17 21:10:51]
    % [X, Y] = MixGauss([[0;0],[1;1]],[0.5,0.25],1000); Y(Y==2)=-1;
    % [k, Vm, Vs, Tm, Ts] = holdoutCVkNN(X, Y, 0.5, 10, intK);
    % errorbar(intK, Vm, sqrt(Vs)); hold on
    % errorbar(intK, Tm, sqrt(Ts)); axis tight;

    nK = numel(intK);

    n = size(X,1);
    ntr = ceil(n*(1-perc));

    Tm = zeros(1, nK);
    Ts = zeros(1, nK);
    Vm = zeros(1, nK);
    Vs = zeros(1, nK);

    ym = (max(Y) + min(Y))/2;

    for rip = 1:nrip

        I = randperm(n);

        % Training set
        Xtr = X(I(1:ntr), :);
        Ytr = Y(I(1:ntr), :);

        % Validation set
        Xvl = X(I(ntr + 1:end), :);
        Yvl = Y(I(ntr + 1:end), :);

        ik = 0;
        for k=intK

            ik = ik + 1;        

            trError =  calcErr(kNNClassify(Xtr, Ytr, k, Xtr), Ytr, ym);
            Tm(1, ik) = Tm(1, ik) + trError;
            Ts(1, ik) = Ts(1, ik) + trError^2;

            valError  = calcErr(kNNClassify(Xtr, Ytr, k, Xvl), Yvl, ym);
            Vm(1, ik) = Vm(1, ik) + valError;
            Vs(1, ik) = Vs(1, ik) + valError^2;

            fprintf('k: %2d, iter: %3d, valErr: %0.3f, trErr: %0.3f\n', k, rip, valError, trError);        
        end
    end

    % Average over trials
    Tm = Tm/nrip;
    Ts = Ts/nrip - Tm.^2;

    Vm = Vm/nrip;
    Vs = Vs/nrip - Vm.^2;

    % Optimum k selection (global minimum of validation error)
    k = min(intK(Vm == min(Vm(:)))); % choose minimum k/model with least complexity
end

function err = calcErr(T, Y, m)
    vT = (T >= m);
    vY = (Y >= m);
    err = sum(vT ~= vY)/numel(Y);
end
