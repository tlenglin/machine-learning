function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%y is a m x 1 matrix, we change it in a m x K matrix, each row = transposed vector [ 0 0 0 ... 1 0 0 ...]

I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :); 
end

%we calculate h_theta = a_3

a_1 = [ones(m, 1) X];

z_2 = a_1 * Theta1'; % m x L = m x (o + 1) * (L * (o + 1))'

a_2 = [ones(size(z_2, 1), 1) sigmoid(z_2)];

z_3 = a_2 * Theta2';%m x K = m x (n + 1) * (K * (n + 1))'

a_3 = sigmoid(z_3); %m x K

%Y = m x K

J = - 1 / m * sum(sum(Y .* log(a_3) + (1 - Y) .* log(1 - a_3), 2));

%adding the regularization, starting from 2 (no regu for the bias term)

J += lambda / (2 * m) * (sum(sum(Theta1(:, 2:size(Theta1, 2)) .^ 2, 2)) + sum(sum(Theta2(:, 2:size(Theta2, 2)) .^ 2, 2)));

%grad = 1 / m * (X' * (sigmoid(X * tmp) - y)) + lambda / m * tmp;
%grad(1) = 1 / m * (X(:,1)' * (sigmoid(X * tmp) - y));


Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));


%we work with trining examples one by one
for t = 1:m

delta_3 = zeros(10, 1);
delta_3 = a_3(t, :)' - Y(t, :)';

tmp = [ones(m, 1) z_2];
delta_2 = Theta2' * delta_3 .* sigmoidGradient(tmp(t, :))';
delta_2 = delta_2(2:end);

Delta_2 = Delta_2 + delta_3 * a_2(t, :);
Delta_1 = Delta_1 + delta_2 * a_1(t, :);

end;

%regularization of the theta gradient
Theta1_grad(:, 1) = 1 / m * Delta_1(:, 1);
Theta2_grad(:, 1) = 1 / m * Delta_2(:, 1);

Theta1_grad(:, 2:end) = 1 / m * Delta_1(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = 1 / m * Delta_2(:, 2:end) + lambda / m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
