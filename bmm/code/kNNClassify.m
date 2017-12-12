function Ypred = kNNClassify(Xtr, Ytr, k, Xte)
    %
    % function Ypred = kNNClassify(Xtr, Ytr, k, Xte)
    %
    % INPUT PARAMETERS
    %   Xtr training input
    %   Ytr training output
    %   k number of neighbours: scalar (odd number suggested)
    %   Xte test input
    %
    % OUTPUT PARAMETERS
    %   Ypred estimated test output
    %
    % EXAMPLE
    %   Ypred = kNNClassify(Xtr, Ytr, 5, Xte);

    n = size(Xtr, 1);

    if k > n
        k = n;
    end

    ylab = unique(Ytr);
    ym = sum(ylab)/2;

    % Center output (if output is not in {-1, 1})
    Ytrm = Ytr - ym;

    % Sort distance matrix column-wise (i.e. for each point in test set)
    [~ ,I] = sort(SquareDist(Xtr, Xte));

    % Read k indices of smallest distances
    idx = I(1:k, :);

    % Prediction, assuming labels have been mapped in {-1,1}
    if k==1;
        % Nearest Neighbor
        Ypred = Ytr(idx);
    else
        val = sum(Ytrm(idx))/k;
        Ypred = sign(val)';
    end

    % m = size(Xte, 1);
    % Ypred = zeros(m, 1);
    % for j = 1:m
    %     % Prediction, assuming labels have been mapped in {-1,1}
    %     val = sum(Ytrm(idx(:,j)))/k;
    %     Ypred(j) = sign(val);
    % end

    % Break ties (k is even) by assigning the closest point label
    indexTie = Ypred==0;
    Ypred(indexTie) = Ytrm(idx(1, indexTie));

    % Map back to original output vals
    Ypred = Ypred  + ym;
end

