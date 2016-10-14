function [trainErrorRate, testErrorRate] = HW2_P1b()

load ('news.mat');

numY = numel(unique(labels));
d = numel(data(1, :));

% first train class prior parameter
prior = hist(labels, unique(labels));
prior = prior / length(labels);

% Then train class conditional distribution parameter 
% size of mu: 20 * 60000
mu = zeros(numY, d);
for y = 1:numY
    idx = find(labels == y);
    colSum = sum(data(idx, :), 1);
    mu(y, :) = (1+colSum) / (2 + length(idx));
end

% training error rate
tmp = (log(mu) * data' + log(1-mu) * (1-data'));
res = bsxfun(@(x,y) x+y, tmp, log(prior'));
[val, idx] = max(res);
trainErrorRate = sum( idx ~= labels') / length(labels);

% test error rate
tmp = (log(mu) * testdata' + log(1-mu) * (1-testdata'));
res = bsxfun(@(x,y) x+y, tmp, log(prior'));
[val, idx] = max(res);
testErrorRate = sum( idx ~= testlabels') / length(testlabels);

end