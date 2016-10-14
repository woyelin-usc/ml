% To call the script, first import 'data', 'labels', 'testdata',
% 'testlabels', then call: HW1_P1(data, labels, testdata, testlabels) in
% matlab Command Window

function stat = HW1_P1(X, Y, testdata, testlabels)
load ('~/Desktop/CU/4771/HW1/ocr.mat')
stat = zeros(4, 1);
for i = 1:10
    idx = 1;
    for n = [1000 2000 4000 8000]
        sel = randsample(length(X), n);
        res = oneNN(X(sel, :), Y(sel), testdata);
        stat(idx) = stat(idx) + sum(res ~= testlabels) / length(testlabels);
        idx = idx+1;
    end
end

stat = stat / 10.0;
errorbar([1000 2000 4000 8000], stat, (mean(stat)-std(stat))*ones(length(stat),1 ));
xlabel('number of selected training set');
ylabel('error tate');
end

function res = oneNN(X, Y, test)

[val, idx ] = min ( (bsxfun(@plus, dot(test,test,2), dot(X,X,2)')- 2 * test*X')' );
res = Y(idx');

end