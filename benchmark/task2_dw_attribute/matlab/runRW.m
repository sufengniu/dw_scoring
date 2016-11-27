function   Pi = runRW(A, option)

% option.step_number
% option.flyout
% option.p
% option.q

N = size(A,1);
d = sum(A,2);
[row, column, value] = find(A);
PG = sparse(row, column, value./(eps+d(row)));
PG(N,N) = 0;
vec = @(x) x(:);

switch option.mode
    
    case 'finite'
                L = option.step_number;
                tmp = PG;
                Pi = PG;
                for i = 1:L-1
                    tmp = PG*tmp;
                    Pi = Pi + tmp;
                end
        
        case 'finite_approx'
                L = option.step_number;
                dims = 20;
                [U S] = eigs(PG, dims, 'LM');
                Uinv = pinv(U);
                s = diag(S);
                s_tmp = s;
                hs = s;
                for i = 1:L-1
                    s_tmp = s_tmp.*s;
                    hs = hs + s_tmp;
                end                
                Pi = max( U*diag(hs)*Uinv, 0);

                
    case 'infinite'
                alpha = option.flyout;
                Pinv = inv( eye(N) - alpha.*PG);
                Pi = ( Pinv - eye(N) )./alpha;
                
                
    case 'infinite_approx'
                alpha = option.flyout;
                dims = 10312;
                
                [U S] = eigs(PG, 20, 'LM');
                Uinv = pinv(U);
                s = diag(S);
                hs = s./(1-alpha.*s);
                Pi = max( U*diag(hs)*Uinv, 0);
                
    case 'memory_finite'
%         p = 10;
%         q = 0.05;
%         PG_d = repmat( PG', [N 1]);
                p =  option.p;
                q =  option.q;
                L = option.step_number;
                PG_diag = sparse(  vec( repmat( (1:N)', [1 N]) ), (1:N^2)', vec( PG ));

                Id = sparse( 1:N, 1:N, 1);
                M = sparse( 1./p.*Id + A + 1./q.*double( double( A^2 ~= 0) - A - Id > 0 ) );
                
%                disp('stage 1: expand')                                    
                T = sparse(N^2, N^2);
                for i =  1:N

                          atmp = repmat( A(i, :), [N 1] );
                          block = (atmp.*M);
                                  
                          block_base = repmat( sum( block, 2)+eps, [1 N]);

                            tmp1 = vec(repmat( ((i-1)*N+1: i*N)', [1 N]) );
                            tmp2 = vec(repmat( i:N:N^2, [N 1]) );
                            tmp3 = vec(block./block_base);
                            T = T + sparse(tmp1, tmp2, tmp3, N^2, N^2 );

                         clear atmp block block_base
                end
       
  %              disp('stage 2: addition')
                tmp = PG_diag;
                Pi_ex = PG_diag;
                clear PG_diag
                for i_step = 2:L
                        tmp = tmp*T;
                        Pi_ex = Pi_ex + tmp;
                end
                Phi = sparse( 1:N^2, repmat( 1:N, [N 1]), 1);
                Pi = Pi_ex*Phi;


    case 'memory_infinite'
            
                p =  option.p;
                q =  option.q;
                alpha = option.flyout;

                Id = sparse( 1:N, 1:N, 1);
                M = sparse( 1./p.*Id + A + 1./q.*double( double( A^2 ~= 0) - A - Id > 0 ) );
                
%                disp('stage 1: expand')                                    
                T = sparse(N^2, N^2);
                for i = 1:N

                          atmp = repmat( A(i, :), [N 1] );
                          block = (atmp.*M);
                                  
                          block_base = repmat( sum( block, 2)+eps, [1 N]);

                            tmp1 = repmat( ((i-1)*N+1: i*N)', [1 N]);
                            tmp2 = repmat( i:N:N^2, [N 1]);
                            tmp3 = vec(block./block_base);
                            T = T + sparse(tmp1, tmp2, tmp3, N^2, N^2 );

                         clear atmp block block_base
                end
                
%                disp('stage 2: inversion')
%                 Id = sparse( 1:N^2, 1:N^2, 1);
%                 tmp = inv( Id - alpha*T);
%                 PG_diag = sparse(  vec( repmat( (1:N)', [1 N]) ), (1:N^2)', vec( PG ));
%                 Pi_ex = PG_diag*tmp;
                
                tmp = sparse(  vec( repmat( (1:N)', [1 N]) ), (1:N^2)', vec( PG ));
                Pi_ex = tmp;
                clear PG_diag
                for i_step = 2:200
                        tmp = alpha.*(tmp*T);
                        Pi_ex = Pi_ex + tmp;
                end
                Phi = sparse( 1:N^2, repmat( 1:N, [N 1]), 1);
                Pi = Pi_ex*Phi;
end
