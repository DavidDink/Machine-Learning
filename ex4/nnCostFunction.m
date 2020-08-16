function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%//NNCOSTFUNCTION Implements the neural network cost function for a two layer
%//neural network which performs classification
%//   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%//   X, y, lambda) computes the cost and gradient of the neural network. The
%//  parameters for the neural network are "unrolled" into the vector
%//   nn_params and need to be converted back into the weight matrices.
%//
%//   The returned parameter grad should be a "unrolled" vector of the
%//  partial derivatives of the neural network.
%//

%//Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
%// for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%// Setup some useful variables
m = size(X, 1);
         
%// You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%// ====================== YOUR CODE HERE ======================
%// Instructions: You should complete the code by working through the
%//               following parts.
%//
%// Part 1: Feedforward the neural network and return the cost in the
%//         variable J. After implementing Part 1, you can verify that your
%//         cost function computation is correct by verifying the cost
%//         computed in ex4.m
%//
%// Part 2: Implement the backpropagation algorithm to compute the gradients
%//         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%//         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%//         Theta2_grad, respectively. After implementing Part 2, you can check
%//         that your implementation is correct by running checkNNGradients
%//
%//         Note: The vector y passed into the function is a vector of labels
%//               containing values from 1..K. You need to map this vector into a
%//               binary vector of 1's and 0's to be used with the neural network
%//               cost function.
%//
%//         Hint: We recommend implementing backpropagation using a for-loop
%//               over the training examples if you are implementing it for the
%//               first time.
%//
%// Part 3: Implement regularization with the cost function and gradients.
%//
%//         Hint: You can implement this around the code for
%//               backpropagation. That is, you can compute the gradients for
%//               the regularization separately and then add them to Theta1_grad
%//              and Theta2_grad from Part 2.
%//

%// Forward Propagate
%// add bias unit and propegate into layer 2
X = [ones(m, 1) X];
z_2 = X * Theta1';
a_2 = [ones(m, 1) sigmoid(z_2)];

%// add bias unit to layer 2 and propegate into output layer
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

%// convert y into a 5000 x 10 matrix with 1's and 0's
y_mat = zeros(m,num_labels);
for i = 1:m,
    y_mat(i,y(i)) = 1;
end;

%// loop through y_mat and a_3 to compute cost for each training example, each are 5000 x 10
for i = 1:m,
J = J + (-1/m)*(  y_mat(i,:)*log(a_3(i,:)') + (1-y_mat(i,:))*log(1 - a_3(i,:)')  );
%//pJ = (-1/m)*(y'*log(h) + (1-y')*log(1 - h)); reference from logistic regression
end;

%// Now add regularization to cost function
Theta1_squared = Theta1.^2; Theta1_squared(:,1) = zeros(hidden_layer_size,1);
Theta2_squared = Theta2.^2; Theta2_squared(:,1) = zeros(num_labels,1);
sum_thetas_squared = sum(sum(Theta1_squared)) + sum(sum(Theta2_squared));
J_reg_term = (lambda/(2*m)) * sum_thetas_squared;

J = J + J_reg_term;

%// Backpropagation
for t = 1:m,
%//forward propagate with one training example
    a_one = X(t,:)';
    z_two = Theta1 * a_one;
    a_two = [1; sigmoid(z_two)];

    z_three = Theta2 * a_two;
    a_three = sigmoid(z_three);

    %// now compute the error for a given training example
    error_3 = a_three - y_mat(t,:)';
    error_2 = (Theta2'* error_3).* sigmoidGradient([1; z_two]);
    error_2 = error_2(2:end);

    Theta1_grad = Theta1_grad + error_2*a_one';
    Theta2_grad = Theta2_grad + error_3*a_two';
end;
           
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

%// Now add the regularization terms for the gradient
Theta1_grad_reg_term = (lambda/m) * Theta1; Theta1_grad_reg_term(:,1) = zeros(hidden_layer_size,1);
Theta2_grad_reg_term = (lambda/m) * Theta2; Theta2_grad_reg_term(:,1) = zeros(num_labels,1);

Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
Theta2_grad = Theta2_grad + Theta2_grad_reg_term;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
