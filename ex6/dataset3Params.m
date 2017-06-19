function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_tab = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_tab = [0.01 0.03 0.1 0.3 1 3 10 30];
results = zeros(length(C_tab) * length(sigma_tab), 3);
count = 1;

for i = 1:length(C_tab)
  for j = 1:length(sigma_tab)
    [model] = svmTrain(X, y, C_tab(i), @(x1, x2) gaussianKernel(x1, x2, sigma_tab(j)));
    predictions = svmPredict(model, Xval);
    results(count, 1) = mean(double(predictions ~= yval));
    results(count, 2) = C_tab(i);
    results(count, 3) = sigma_tab(j);
    %mean(double(predictions ~= yval))
    %C_tab(i)
    %sigma_tab(j)
    count += 1;
  end;
end;  

[value, index] = min(results(:, 1));
C = results(index, 2);
sigma = results(index, 3);

%value
%index
%C
%sigma

% =========================================================================

end
