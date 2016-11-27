 function   [Y  Uw mu] = weightedGraph(A, X, option)

% load('/Users/sihengc/Documents/Research/Project/Graph_Signal_Processing/Toolbox/Social_Science/embedding/software/deepwalk-master/example_graphs/blogcatalog.mat')
% N = size(network, 2);
% % order = randperm(N);
% A = network; %(order, order);
% X = group; %(order, :);
% z = X(:, 13);
% z(find(z~=1)) = -1;

alpha = option.flyout;
% alpha = 0.85;

    [N T] = size(X);

    d = sum(A,2);
    [row, column, value] = find( A);
    PG = sparse(row, column, value./(eps+d(row)));
    PG(N,N) = 0;
    mu = 0;
    
if T == 0

            alpha = option.flyout;
            Pinv = inv( eye(N) - alpha.*PG);
            Y = log( ( Pinv - eye(N) )./alpha + eps^2);

else

            for i_att = 1:T
                    PA = X(:, i_att)*X(:, i_att)'./(eps+sum(X(:, i_att)));             
                    flyout_ind = find( X(:, i_att) == 0 );     
                    PA( flyout_ind, :) = PG( flyout_ind,:);
                    P_Ai{i_att} = PA;     
                    clear tmp Xc PA
            end


            % Z template            

        switch option.weight

            case 'attribute'
                
                        for i_att = 1:T
                                PA = X(:, i_att)*X(:, i_att)'./(eps+sum(X(:, i_att)));
                                P_Ai{i_att} = PA;     
                                clear tmp Xc PA
                        end
                        mu  = ones(T,1)./T;
                        P = sparse(N,N);
                        for i = 1:length(P_Ai)
                            P = P + mu(i).*P_Ai{i};
                        end
                        Pinv = inv( eye(N) - alpha.*P);
                        Y = log( ( Pinv - eye(N) )./alpha + eps^2);
                
            case 'equal'

                        mu  = ones(T,1)./T;
                        Pinv = mergeTransition( mu, PG, P_Ai, alpha );
                        Y = log( ( Pinv - eye(N) )./alpha + eps^2);
                        
            case 'one-shot'

                        Z = option.label_template; 
                        for i_att = 1:T
                               mu(i_att,1) = sum( sum( Z.*( P_Ai{i_att} - PG ) ) );
                        end
                        [value index] = sort(mu, 'descend');

                        if option.sparse == 1
                            mu( mu < value(6)) = 0;
                        end

                        mu = max(mu, 0);
                        mu = mu./( eps+sum(mu) );
                        if sum(mu ~= 0)
                                mag_set = [0  0.25  0.5  0.75 1];
                                for i_mag = 1:length(mag_set)
                                    Pinv = mergeTransition( mag_set(i_mag).*mu, PG, P_Ai, alpha );
                                    obj2(i_mag,1) = sum( sum( Z.*log( ( Pinv - eye(N) )./alpha + eps^2) ));
                                    clear Pinv;
                                end
                                [~, index] = max(obj2);
                                mu = mag_set(index).*mu;
                        end
                        Pinv = mergeTransition( mu, PG, P_Ai, alpha );
                        Y = log( ( Pinv - eye(N) )./alpha + eps^2);

                case 'gradient descent'

                        Z = option.label_template ; 
                        mu  = (ones(T,1)-0.5)./T;
                        for i_iter = 1:5

                                mu_set(:, i_iter) = mu;
                                Pinv = mergeTransition( mu, PG, P_Ai, alpha );
                                obj(i_iter,1) = sum( sum( Z.*log( ( Pinv - eye(N) )./alpha + eps^2) ));
                                tmp = Z./(Pinv - eye(N));

                                for i_mu = 1:T

                                        C  = sum(X(:, i_mu));
                                        C_index = find( X(:, i_mu) );
                                    tic;  
                                        tmp1 = sparse( ones(C , 1)*X(:, i_mu)'./C ) - PG( C_index, :);
                                        tmp2 = Pinv(:, C_index  )*tmp1;
                                        tmp3 = tmp2*Pinv;
                                        mu_gradient(i_mu, 1) = sum( sum( tmp.* tmp3 ) );
                                    toc;
                                end
                                mu = max( mu + 0.1.*norm(mu, 2)/norm(mu_gradient, 2).*mu_gradient, 0);         
                        end

                        [~, index] = max(obj);
                        mu = mu_set(:, index);
                        Pinv = mergeTransition( mu, PG, P_Ai, alpha );
                        Y = log( ( Pinv - eye(N) )./alpha + eps^2);

        end
end
    [U S V] = svds(Y, option.dimension);
    Uw = U*diag( sqrt(diag(S)) );
    %Objective = sum( sum( option.label_template.*Y) );
    