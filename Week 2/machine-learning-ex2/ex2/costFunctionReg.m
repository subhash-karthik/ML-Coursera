function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n= length(theta);

% You need to return the following variables correctly 
J = 0;
l=lambda.*(ones(n,1)-[1;zeros(n-1,1)]);
grad = zeros(size(theta));
J=sum((1-y).*log(1-sigmoid(X*theta)))+sum(y.*log(sigmoid(X*theta )));
J=-1*J/m;
J=J+sum(theta(2:n).^2)*lambda/(2*m);
grad=sum(repmat(sigmoid(X*theta)-y,1,size(X)(2)).*X)./m;
grad=grad+transpose((l.*theta)./m);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end