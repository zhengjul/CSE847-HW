% Random data generation
sigma = eye(2)*10;
n = 1000;
df1 = mvnrnd([-1, -10], sigma, n);
df2 = mvnrnd([-10, 1], sigma, n);
df3 = mvnrnd([10, 1], sigma, n);
df4 = mvnrnd([12, 12], sigma, n);
data = [df1;df2;df3;df4];

colormap(copper(5))
subplot(2,2,1);
scatter1 = scatter(data(:,1), data(:,2),[]);
alpha(scatter1,.2);
title("Random Dataset");

%% k-means cluster where k = 4 for our dataset
kmeans_result = k_means(data, 4);

subplot(2,2,2);
scatter2 = scatter(data(:,1), data(:,2), [], kmeans_result);
alpha(scatter2,.2);
title("k-means, k=4");


%% spectral relaxed k-means, k = 4, sigma = 1
spectral_kmeans_result = spectral_relaxed_kmeans(data, 4, 1);
subplot(2,2,3);
scatter4 = scatter(data(:,1), data(:,2), [], spectral_kmeans_result);
alpha(scatter4,.2);
title("spectral k-means, k=4, sigma=1");


%% k-means algorithm
function [clustered_datapoints] = k_means(data, K)
    [n, d] = size(data);
    clustered_datapoints = zeros(n, 1);
    
    % initialize centroids
    centroids = rand(K,d);
    old_centroids = rand(K,d) * 10;
    
    % stopping condition: centroid stops shifting around
    while norm(centroids - old_centroids) > 0.0001
        for i=1:n
            point = data(i,:);
            dist = sqrt(sum(bsxfun(@minus, point, centroids).^2,2));  
            clustered_datapoints(i) = find(dist==min(dist)); % cluster point to closest centroid
        end
        
        old_centroids = centroids;
        for i=1:K
            datapoints_in_cluster = zeros(0,d);
            for j=1:n  % get all the points for this cluster i
                point = data(j,:);
                if clustered_datapoints(j) == i
                    datapoints_in_cluster = [datapoints_in_cluster ; point];
                end
            end
            centroids(i,:) = mean(datapoints_in_cluster, 1);
        end
    end
end
%% Spectral relaxation for k-means
function [clustered_datapoints] = spectral_relaxed_kmeans(data, K, sigma)
    % sigma is used in the Gaussian similarity function s_ij such that
    % sigma controls the width of the neighbourhoods in the fully connected
    % graph (connect all points with positive similarity with each other)
    
    % Laplacian is normalized with paper:
    %Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. IEEE Transactions on pattern analysis and machine intelligence, 22(8), 888-905.
    
    [N, d] = size(data);
    clustered_datapoints = zeros(N, 1);
    
    % Matrix A (Adjacency matrix) of data vectors with associated sum-of-squares cost
    % function 
    matrix_distances = squareform(pdist(data, 'euclidean'));
    

    matrix_A = exp(-(matrix_distances./sigma).^ 2);
    diag(matrix_A)
    size(matrix_A)
    % Degree matrix D is the trace of matrix A (a trace is the sum of row elements placed, result written on the main diagonal)
    %matrix_D = trace(matrix_A);
    matrix_D = diag(sum(matrix_A, 2));
    size(matrix_D)
    
    matrix_L = matrix_D - matrix_A;  % Un-normalized Laplacian
    matrix_L_norm = inv(matrix_D)*matrix_L; % Shi & Malik (2000) normalized Laplacian matrix
    
    % sort eigenvectors by ascending eigenvalues and extract K smallest
    % eigenvalues' eigenvectors and put into matrix X
    [eigenvectors, eigenvalues] = eigs(matrix_L_norm, K,'smallestabs','FailureTreatment','drop'); % drop NAs
    eigenvalues
    % X is n by k orthonormal matrix corresponding to k-smallest
    % eigenvalues of the Laplacian matrix
    matrix_X = eigenvectors ;
    disp(size(matrix_X));
    
    % After data has been transformed, apply ordinary k-means
    clustered_datapoints = kmeans(matrix_X, K);
end