function [weights] = logistic_train(data, labels, epsilon, maxiter)
%% HW4 Q1: Newton-Raphson (IRLS) iterative reweighted least squares (gradient descent)
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%
weights = zeros(size(data,2),1);
if nargin == 3
    maxiter=1000;
elseif nargin < 3
    epsilon=1e-5;
    maxiter=1000;
end
for i = 1:maxiter
    y = sigmf(data * weights, [1 0]); % predicted y can be 0/1
    R = diag(y.*(1 - y));
    noise = 0.01;   %my R matrices were close to singular so I'm adding this noise to make it not singular
    R = R + noise * eye(length(R));
    z = data * weights - inv(R)*(y - labels);
    
    weights = inv(data'*R*data)*data'*R*z;
    
    y_updated = sigmf(data * weights, [1 0]);
    
    if mean(abs(y_updated - y)) < epsilon    %if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
       break;
    end
    
end

end


