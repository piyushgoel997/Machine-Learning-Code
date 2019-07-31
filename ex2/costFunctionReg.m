
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

% cost function without regularisation
hTheta = theta'*X';
hTheta = (sigmoid(hTheta))';
J = -(y'*log(hTheta) + (1-y)'*log(1-hTheta));
J = J/m;

% theta0 can now be set equal to zero in the calcutlation of regTerm for
% ease
theta(1) = 0;
regTerm = theta'*theta;
regTerm = (lambda/(2*m))*regTerm;
J = J + regTerm;


grad = (X'*(hTheta-y))/m + (lambda/m)*theta;

% =============================================================

end
