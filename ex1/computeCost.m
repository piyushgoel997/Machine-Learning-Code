function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


hTheta = X*theta;

sumMatrix = hTheta(:,1)-y(:,1);

sum = 0;

for index = 1:m
  sum = sum + sumMatrix(index,1)*sumMatrix(index,1);
end

J = sum/(2*m);

% =========================================================================

end
