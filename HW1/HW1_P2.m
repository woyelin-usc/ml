% function res = HW1_P2 (X, Y)
% m = num2cell([1000 2000 4000 8000]);
% res = HW1_P2_tmp(X, Y, m[:] )
% end

function stat = HW1_P2(X, Y, m)
load ('~/Desktop/CU/4771/HW1/ocr.mat');

stat = zeros(10,1);
for times = 1:10
    times
    % call prototype selection method to get the selected prototypes
    [data1, labels1] = ps(X, Y, m);
    % evaluate the test error rate over testdata
    res = oneNN(data1, labels1', testdata);
    stat(times) = sum( res ~= testlabels) / length(testlabels);
end
end

function [data1, labels1] = ps(data, labels, m)
    data1 = ([]);
    labels1 = ([]);
    count = 0;
    for i=randperm(length(data))
        if length(data1) == 0;
            data1 = [data1; data(i, :)];
            labels1 = [labels1 labels(i)];
            count = count+1;
            if count == m
                break;
            end
        elseif oneNN(data1, labels1, data(i, :)) ~= labels(i);
            data1 = [data1; data(i,:)];
            labels1 = [labels1 labels(i)];
            count = count+1;
            if count == m
                break;
            end
        end
    end

    % randomly select m-length(data1) to make sure we have right number of
    % prototypes
    if m-length(data1) > 0
        sel = randsample(length(data), m-length(data1));
        data1 = [data1; data(sel, :)];  
        labels1 = [labels1 labels(sel)'];
    end
end


function res = oneNN(X, Y, test);

[val, idx ] = min ( (bsxfun(@plus, dot(test,test,2), dot(X,X,2)')- 2 * test*X')' );
res = Y(idx');

end