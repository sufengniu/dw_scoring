
function  [Uw Vw Y ]= obtainEmb(Pi, option)


switch option.wrap
    case 'linear'
        Y = Pi;
        
    case 'poly'
        
                power = option.poly_power;
                if power > 0
                    Y =  1/power.*( Pi.^power );  
                elseif power == 0
                    Y = log( Pi + eps^2 );
                else
                    Y = 1/power.*(Pi +eps^4).^power;  
                end
        
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

