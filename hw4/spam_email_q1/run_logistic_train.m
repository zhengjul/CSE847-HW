load data.txt
load labels.txt

data = [ones(size(data,1),1) , data]; % add intercept term of dummy variable 1's
x_train = data(1:2000,:);
y_train = labels(1:2000,:);
x_test = data(2001:4601,:);
y_test = labels(2001:4601,:);
test_size = size(y_test,1);

training_samples = [200,500,800,1000,1500,2000];
n_tests = size(training_samples,2);
accuracy = zeros(n_tests,1);
%% Train your logistic regression classifier on the first n rows of the training data
for i = 1:n_tests
    n = training_samples(i)
    x_train_n = x_train(1:n,:);
    y_train_n = y_train(1:n,:);
    weights = logistic_train(x_train_n, y_train_n);
    
    predictions = round(sigmf(x_test*weights,[1 0]));
    accuracy(i) = sum(y_test == predictions)/test_size;
end

figure;
plot(training_samples, accuracy, 'mo-');
title('Q1 Logistic Regression');
xlabel('n (training data size)');
ylabel('Testing Accuracy');