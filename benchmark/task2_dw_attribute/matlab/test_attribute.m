clear
clc

addpath('libsvm');
% parpool


load('../dataset/kaggle/1968.mat')
A = network; 
Target = location;
X = double( education ~=0 );   % location_id
m = size(X,2);

[N T] = size(Target);
LRatio_set = 0.5; %[0.1 0.5 0.9];

option.step_number = 7;
option.flyout = 0.85;
option.dimension = 12;
option.p = 0.5;
option.q = 0.5;
option.mode = 'infinite';
option.wrap = 'exp';
option.emb = 'svd';

 Pi = runRW(A, option);
[Uw{1} Vw Y ]= obtainEmb(Pi, option);



for i_LR = 1:length(LRatio_set)
    
    LRatio = LRatio_set(i_LR);
        
for i_cross = 1:10

    for i_target = 1:T

i_target, i_LR, i_cross

                    target = Target(:, i_target);
                    label = target;
                    label(label ~=1) = -1;

                    [Test, Train] = crossvalind('LeaveMOut', N, round( LRatio*N));
                    Train_Ind = find(Train);
                    Train_Ind_pos = Train_Ind( find( label(Train, :) == 1));
                    Train_Ind_neg = Train_Ind( find( label(Train, :) == -1));


                    template = sparse(N,1);
                    template(Train,:) = target(Train, :);
                    Z = double(template*template' > 0);
                    Z(find(Z ==0)) = -1;
                    zero_ind = find( sum( template, 2) == 0);
                    Z(zero_ind, zero_ind) = 0;
                    pos_num = sum(sum(Z== 1));
                    neg_num = sum(sum(Z==-1));
                    Z( Z ==-1) = -pos_num/neg_num;
                    option.label_template = Z;


                     % non-overlapping
                     option.sparse = 0;
                     option.weight = 'one-shot';
                     coding  = logic_fit( target(Train, :), X(Train, :), 5);
                     [Y  Uw{2} mu] = weightedGraph(A,  [X(:, find(coding) ) double( X*coding > 10^-6 )], option);

                for i = 1:3
                    
                    if i < 3
                            U0 = Uw{i};
                            pos_Ind = find( label(Train_Ind) == 1 );
                            neg_Ind = find( label(Train_Ind) == -1 );
                            [T1, T2] = crossvalind('LeaveMOut', length(neg_Ind), min( length(neg_Ind), max( round(length(pos_Ind)), 1) ) );
                            train_ind = Train_Ind( [neg_Ind(T2); pos_Ind]);            

                            model_linear = svmtrain(  label(train_ind), sparse( U0(train_ind, :) ), '-t 0' );
                            test_preds = svmpredict( label(Test), sparse( U0(Test, :) ), model_linear); 
                    else
                             coding  = logic_fit( target(Train, :), X(Train, :), 1);
                            test_preds =  X(Test, find(coding) );
                    end
                    
                         C =  find( label(Test,1)==1 );
                    Cres =  find( test_preds==1 );
                    TP(i_target, i)  =   length( intersect(C, Cres) );
                    precision_de(i_target, i) = length( Cres )+eps;
                    recall_de(i_target, i) = length( C) +eps;

                    precision = (length( C ) == 0)*1  + (length( C ) ~= 0)*length( intersect(C, Cres) )/(eps+ length( Cres));
                    recall =  (length( C ) == 0)*1  + (length( C ) ~= 0)*length( intersect(C, Cres) )/(eps+length( C ));
                    f1(i_target, i) = 2/(1/precision + 1/recall );
                    
                end
        end
        
            Macro_F1( :, i_LR, i_cross) = mean(f1);
            pre= sum(TP,1)./sum(precision_de,1);
            re = sum(TP,1)./sum(recall_de,1);
            Micro_F1( :, i_LR, i_cross) = 2./(1./pre + 1./re );
            
            clear train_best TP precision_de recall_de precision recall

end
end

mean(Micro_F1, 3)
mean(Macro_F1, 3)



obj_fun = @(x,y) sum( y'*x )/( sum(y) + sum( x ) - sum( y'*x ) )';
clear corr
for i = 1:size(Target,2)
for j = 1:size(X,2)
    corr(i,j) = obj_fun( double( Target(:, i)~=0 ),  double( X(:, j)~=0) );
end
end
imagesc(corr)