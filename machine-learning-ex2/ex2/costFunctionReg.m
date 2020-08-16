function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
error = h - y;

%// Initialize pseudo J and grad to non-regularized regression
pJ = (-1/m)*(y'*log(h) + (1-y')*log(1 - h));
pgrad = (1/m)*(X'*error);
               
%// compute terms specific to regularization
theta_squared = theta.^2;
J_reg_term = (lambda/(2*m)) * ( sum(theta_squared) - theta_squared(1) );
grad_reg_term = (lambda/m) * theta; grad_reg_term(1) = 0;

%// return J and grad
J = pJ + J_reg_term;
grad = pgrad + grad_reg_term;

% =============================================================

end
