function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

z = X * theta;
P = sigmoid (z);
% s = size (P);
% % Compute the cost term for y = 1
% A = log (P);
% % Compute the cost term for y = 0
% B = log (ones(s) - P);
% % Debugging matrix size
% % sizeB = size(B)
% % Combine into one term
% C = y .* A + (ones(m) - y) .* B;
% Debugging matrix size
% sizeC = size(C)
% Take sums and compute final cost function

J = (-1 / m) * sum(y.*log(sigmoid(X * theta)) + (1 - y).*log(1 - sigmoid(X * theta)));

%%% Derivatives or gradient

grad = ((P - y)' * X )' ./ m;


% =============================================================

end
