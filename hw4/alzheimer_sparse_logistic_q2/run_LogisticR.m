load ad_data.mat
load feature_name.mat

%% different l1 regularization parameters to try
par = [0,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
num_par = size(par,2);
accuracy = zeros(num_par,1); % accuracies
AUC_list = zeros(num_par,1);      % areas under curves
num_features_list = zeros(num_par,1);
for i = 1:num_par
    % for each parameter, train l1-norm logistic regression and see its
    % performance on the test dataset given
    [weights, bias] = logistic_l1_train(X_train, y_train, par(i)); %training set
    num_features_list(i) = sum(weights ~= 0); %non-zero weights represent selected features (AKA useful)
    
    % testing model and recording AUC
    predictions = X_test * weights + bias; % testing set   
    [X, Y, T, AUC] = perfcurve(y_test, predictions, 1); %(X,Y) are coordinates of ROC curve, T is thresholds for computed X and Y, AUC is area under the curve for the computed X and Y
    AUC_list(i) = AUC;
    
    
    
    
end

figure;
plot(par, AUC_list, 'm*-');
title('Q2 Sparse Logistic Regression Experiment');
xlabel('l1 Regularization Parameter');
ylabel('Area Under Curves');


figure;
plot(par, num_features_list, 'mo-');
title('Q2 Sparse Logistic Regression Experiment');
xlabel('l1 Regularization Parameter');
ylabel('Number of Selected Features (non-zero weights)');





function [w, c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations.
[w, c] = LogisticR(data, labels, par, opts);
end