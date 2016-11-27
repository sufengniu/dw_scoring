
function coding  = logic_fit( y, X, k )


[N T] = size(X);
obj_fun = @(tmp) sum( y'*tmp )/( sum(y) + sum( tmp ) - sum( y'*tmp ) )';  

x_aug = sparse(N, 1);

coding = sparse(T, 1);

for i_k = 1:k
    
    for i = 1:T
        tmp =  double( x_aug + X(:,i) > 0 );
        Corr(i,1) = obj_fun(tmp);
%         tmp =  double( x_aug - X(:,i) > 0 );
%         Corr(i,2) = obj_fun(tmp);
    end
    [value index] = max(Corr(:));
    
    Index = mod(index, T);
    if Index == 0
        Index = T;
    end
    tmp = double( ( x_aug + (index <= T)* X(:, Index ) - (index > T)* X(:, Index ) ) > 0 );
    
    if obj_fun(tmp) > obj_fun(x_aug)
            x_aug = tmp;
            Index = mod(index, T);
            if Index == 0
                Index = T;
            end
            coding( Index , 1) = double(index <= T) - double(index > T);
    end
 
end