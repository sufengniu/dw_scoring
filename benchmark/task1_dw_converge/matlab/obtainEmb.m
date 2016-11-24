
function  [Uw Vw Y ]= obtainEmb(Pi, option)


switch option.wrap
    case 'linear'
        Y = Pi;
        
    case 'poly'
        Y = Pi.^(0.01);  
        
    case 'exp'
        Y = log( Pi + eps^2 );

end


switch option.emb
    case 'svd'
        [U S V] = svds(Y, option.dimension);
        Uw = U*diag( sqrt(diag(S)) );
        Vw = V*diag( sqrt(diag(S)) );
        
    case 'cross_entropy'
       [Uw, Vw] = CrossEntropyDecomp(Y, option.dimension);

    case 'correlation'
        [Uw, Vw] = CorrelationDecomp( Y, option.dimension);
end

