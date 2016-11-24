
% min  || M - X Y' ||_F^2 
% s.t.   Y = log X

%  L(X, Y, Mu) = || M - X Y' ||_F^2  + lambda/2 || Y - log X - Mu/lambda||_F^2

function [X Y] = CrossEntropyDecomp( M, dim )

% N = 100;
% dim = 5;
% X0 = rand(N,dim);
% M = X0*log(X0)';

N = size(M,1);
[U S V] = svds(M, dim);
X = max( U*diag( sqrt(diag(S)) ), eps^2);
Y = V*diag( sqrt(diag(S)) );

Mu = zeros(N,dim);
step_size = 0.01;
lambda = 0.1;

norm( Y - log(X), 'fro')
norm( M - X*Y', 'fro')^2

obj_fun = @(X,Y)  norm( M - X*Y', 'fro')^2  + norm(Y - log(X) - Mu./lambda, 'fro')^2*lambda/2;
Y_gradient = @(X,Y) -2.*(M-X*Y')'*X + lambda.*(Y-log(X)-Mu./lambda);
X_gradient = @(X,Y) -2.*(M-X*Y')*Y - lambda.*( (Y-log(X)-Mu./lambda)./X );

for i_ter = 1:500
tic;
    tmp = Y_gradient(X,Y);
    Y = Y - (step_size*norm(Y,'fro')/norm(tmp,'fro')).*tmp;
    
    tmp = X_gradient(X,Y);
    X = X - (step_size*norm(X,'fro')/norm(tmp,'fro')).*tmp;
    
    X = max( X, eps^2 );
    Mu = Mu - lambda.*(Y - log(X) );
    obj_val(i_ter,1) = obj_fun(X,Y);
 toc;
end

norm( Y - log(X), 'fro')
norm( M - X*Y', 'fro')^2

Y = exp(Y);