function Y = flipLabels(Y, p)
    % 
    % Yn = flipLabels(Y, p) flips p percent of the labels in Y. Labels are 
    % assumed to be {+1, -1}.

    n = numel(Y);
    n_flips = floor(n*p);

    sel = randperm(n, n_flips);

    Y(sel) = -1*Y(sel);
end