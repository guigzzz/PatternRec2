%% Preprocessing
%rng(1);
data = importdata('wine.data.csv', ' ');
split = data(:, 1);
labels = data(:, 2);
all_data = data(:, 3:end);

train_set = all_data(split == 1, :);
train_labels = labels(split == 1);
test_set = all_data(split == 2, :);
test_labels = labels(split == 2);

%{
normalised_all_data = zeros(size(all_data));
for i = 1:size(all_data, 2)
    normalised_all_data(:, i) = all_data(:, i) ./ norm(all_data(:, i));
end

normalised_all_data = all_data ./ max(all_data, [], 1);
G = chol(inv(cov(all_data)));
normalised_G = chol(inv(cov(normalised_all_data)));

normalised_train_set = normalised_all_data(split == 1, :);
normalised_test_set = normalised_all_data(split == 2, :);
%}

train_std = std(train_set);
normalised_train_set = train_set ./ train_std;
normalised_test_set = test_set ./ train_std;

G = chol(inv(cov(train_set)));
normalised_G = chol(inv(cov(normalised_train_set)));

%% Q1
fprintf('####### Q1 ########\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

metrics = ["cityblock" "euclidean" "chebychev" "cosine" "mahalanobis" "correlation"];
chisquare = @(x, y) sqrt(0.5 * sum((x - y) .^ 2 ./ (x + y), 2));
intersection = @(x, y) 1 - 0.5 * (sum(min(x, y), 2) ./ sum(x) + sum(min(x, y), 2) ./ sum(y, 2));

%%%%%%% NO NORMALISATION %%%%%%%%%
for i = 1:size(metrics, 2)
    idx = knnsearch(train_set, test_set, 'Distance', char(metrics(i)));
    acc = 1 - mean(test_labels == train_labels(idx));
    fprintf('raw, %s: %f\n', metrics(i), acc);
end

idx = knnsearch(train_set, test_set, 'Distance', chisquare);
acc = 1 - mean(test_labels == train_labels(idx));
fprintf('raw, root chi square: %f\n', acc);

idx = knnsearch(train_set, test_set, 'Distance', intersection);
acc = 1 - mean(test_labels == train_labels(idx));
fprintf('raw, intersection: %f\n', acc);

fprintf('\n');
%%%%%%% NORMALISATION %%%%%%%%%
for i = 1:size(metrics, 2)
    idx = knnsearch(normalised_train_set, normalised_test_set, 'Distance', char(metrics(i)));
    acc = 1 - mean(test_labels == train_labels(idx));
    fprintf('normalised, %s: %f\n', metrics(i), acc);
end

idx = knnsearch(normalised_train_set, normalised_test_set, 'Distance', chisquare);
acc = 1 - mean(test_labels == train_labels(idx));
fprintf('normalised, root chi square: %f\n', acc);

idx = knnsearch(normalised_train_set, normalised_test_set, 'Distance', intersection);
acc = 1 - mean(test_labels == train_labels(idx));
fprintf('normalised, intersection: %f\n', acc); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Q2
fprintf('\n####### Q2 ########\n');
%%%%%%%%%%%%%%%%%%%%%% KMEANS %%%%%%%%%%%%%%%%%%%%%%%%%%%

metrics = ["cityblock" "sqeuclidean" "cosine" "correlation"];
nnmetrics = ["cityblock" "euclidean" "cosine" "correlation"];
k = 3;
c_labels = zeros(k, 1);
%%%%%%% NO NORMALISATION %%%%%%%%%
%{
for i = 1:size(metrics, 2)
        [~, c, ~, transformed_train_set] = kmeans(train_set, k, 'Distance', char(metrics(i)), 'Replicates', 10);
        [~, ~, ~, transformed_test_set] = kmeans(test_set, k, 'Distance', char(metrics(i)), 'Start', c, 'MaxIter', 1);

        idx = knnsearch(transformed_train_set, transformed_test_set, 'Distance', char(nnmetrics(i)));
        acc = 1 - mean(test_labels == train_labels(idx));
        fprintf('raw, %s: %f\n', metrics(i), acc);
end
%}

for i = 1:size(metrics, 2)
        [cluster_idx, c] = kmeans(train_set, k, 'Distance', char(metrics(i)), 'Replicates', 10);
        for j = 1:k
            c_labels(j) = mode(train_labels(cluster_idx == j));
        end

        knn_idx = knnsearch(c, test_set, 'Distance', char(nnmetrics(i)));
        acc = 1 - mean(test_labels == c_labels(knn_idx));
        fprintf('raw, %s: %f\n', metrics(i), acc);
end


[~, c, ~, transformed_train_set] = kmeans(train_set, k, 'Distance', 'sqeuclidean');
[~, ~, ~, transformed_test_set] = kmeans(test_set, k, 'Distance', 'sqeuclidean', 'Start', c, 'MaxIter', 1);
idx1 = knnsearch(transformed_train_set, transformed_test_set, 'Distance', 'mahalanobis');
acc = 1 - mean(test_labels == train_labels(idx1));
fprintf('raw, mahalanobis: %f\n', acc);


%{
[cluster_idx, c] = kmeans(train_set, k, 'Distance', 'sqeuclidean', 'Replicates', 10);
for j = 1:k
    c_labels(j) = mode(train_labels(cluster_idx == j));
end
knn_idx = knnsearch(c, test_set, 'Distance', 'mahalanobis');
acc = 1 - mean(test_labels == c_labels(knn_idx));
fprintf('raw, mahalanobis: %f\n', acc);
%}

fprintf('\n');
%%%%%%% NORMALISATION %%%%%%%%%
%{
for i = 1:size(metrics, 2)
    [~, c, ~, transformed_train_set] = kmeans(normalised_train_set, k, 'Distance', char(metrics(i)), 'Replicates', 10);    
    [~, ~, ~, transformed_test_set] = kmeans(normalised_test_set, k, 'Distance', char(metrics(i)), 'Start', c, 'MaxIter', 1);

    idx = knnsearch(transformed_train_set, transformed_test_set, 'Distance', char(nnmetrics(i)));
    acc = 1 - mean(test_labels == train_labels(idx));
    fprintf('normalised, %s: %f\n', metrics(i), acc);
end
%}

for i = 1:size(metrics, 2)
    [cluster_idx, c] = kmeans(normalised_train_set, k, 'Distance', char(metrics(i)), 'Replicates', 10);
    for j = 1:k
        c_labels(j) = mode(train_labels(cluster_idx == j));
    end

    knn_idx = knnsearch(c, normalised_test_set, 'Distance', char(nnmetrics(i)));
    acc = 1 - mean(test_labels == c_labels(knn_idx));
    fprintf('normalised, %s: %f\n', metrics(i), acc);
end


[~, c, ~, transformed_train_set] = kmeans(normalised_train_set * normalised_G, k, 'Distance', 'sqeuclidean');
[~, ~, ~, transformed_test_set] = kmeans(normalised_test_set * normalised_G, k, 'Distance', 'sqeuclidean', 'Start', c, 'MaxIter', 1);
idx = knnsearch(transformed_train_set, transformed_test_set, 'Distance', 'mahalanobis');
acc = 1 - mean(test_labels == train_labels(idx));
fprintf('normalised, mahalanobis: %f\n', acc);


%{
[cluster_idx, c] = kmeans(normalised_train_set, k, 'Distance', 'sqeuclidean', 'Replicates', 10);
for j = 1:k
    c_labels(j) = mode(train_labels(cluster_idx == j));
end
knn_idx = knnsearch(c, normalised_test_set, 'Distance', 'mahalanobis');
acc = 1 - mean(test_labels == c_labels(knn_idx));
fprintf('normalised, mahalanobis: %f\n', acc);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%