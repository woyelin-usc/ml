function [trainErrorRate, testErrorRate, a0, a, largeWords, smallWords] = HW2_P1cd()

load ('news.mat');

newdata1 = data(find(labels==1 | labels==16 | labels==20), :);
newlabels1 = zeros(length(newdata1(:, 1)), 1);
newdata2 = data(find(labels==17 | labels==18 | labels==19), :);
newlabels2 = ones(length(newdata2(:,1)), 1);

newdata = [newdata1; newdata2];
newlabels = [newlabels1; newlabels2];


newtestdata1 = testdata(find(testlabels==1 | testlabels==16 | testlabels==20), :);
newtestlabels1 = zeros(length(newtestdata1(:, 1)), 1);
newtestdata2 = testdata(find(testlabels==17 | testlabels==18 | testlabels==19), :);
newtestlabels2 = ones(length(newtestdata2(:,1)), 1);

newtestdata = [newtestdata1; newtestdata2];
newtestlabels = [newtestlabels1; newtestlabels2];

% first train class prior parameter
% prior:
prior = [ sum(newtestlabels==0) / length(newtestlabels) ];
prior = [ prior 1-prior(1)];

% then train class conditional distribution
mu = zeros(2, length(newdata(1,:)));

idx = find( newlabels==0 );
colSum = sum( newdata(idx, :), 1);
mu(1, :) = (1+colSum) / (2 + length(idx) );

idx = find( newlabels==1 );
colSum = sum( newdata(idx, :), 1);
mu(2, :) = (1+colSum) / (2+length(idx));

% training error rate
tmp = (log(mu) * newdata' + log(1-mu) * (1-newdata'));
res = bsxfun(@(x,y) x+y, tmp, log(prior'));
[val, idx] = max(res);
trainErrorRate = sum( idx-1 ~= newlabels') / length(newlabels);

% test error rate
tmp = (log(mu) * newtestdata' + log(1-mu) * (1-newtestdata'));
res = bsxfun(@(x,y) x+y, tmp, log(prior'));
[val, idx] = max(res);
testErrorRate = sum( idx-1 ~= newtestlabels') / length(newtestlabels);


% This is for problem 2 (d)
a0 = log (prior(2)) - log (prior(1)) + sum( log(1-mu(2 ,:)) - log (1-mu(1, :)));
a = log (mu(2, :)) - log(mu(1, :)) - log(1-mu(2, :)) + log(1-mu(1, :));
% a = log ( (mu(2, :) .* (1-mu(1, :))) ./ ( mu(1,:).*(1-mu(2, :)))  );
[v, i] = sort(a);

large = i(end:-1:end-19);
small = i(1:20);

% read vocabulary line by line
fid = fopen('./news.vocab');
tline = fgetl(fid);
vocab = {};
i = 1;
while ischar(tline)
    word{i} = tline;
    i=i+1;
    tline = fgetl(fid);
end
fclose(fid) = fopen('./news.vocab');
size(vocab);

% 20 largest words
largeWords = word(large);
% 20 smallest words
smallWords = word(small);

end