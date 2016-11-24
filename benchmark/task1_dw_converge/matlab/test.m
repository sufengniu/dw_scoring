
files = dir('/Users/sihengc/Documents/Research/Project/Graph_Signal_Processing/Toolbox/Social_Science/embedding/dw_scoring/benchmark/task1_dw_converge/data/kaggle/*.mat');
filename = 'kaggle/4406'; % 'Flickr'  'blogcatalog';
load(['../data/' filename '.mat'])

A = network;

Walk = {'finite', 'infinite'}
Wrap = {'exp', 'tan'};
Emb = {'svd', 'cross_entropy'}
option.dimension = 12;
option.step_number = 7;
option.flyout = 0.85;
        
        option.mode = 'infinite';
        Pi = runRW(A, option);
        
        option.wrap = 'exp';
        option.emb = 'svd';   %   svd      cross_entropy
        [Uw Vw Y]= obtainEmb(Pi, option);
        
        
        embedding = [Uw Vw]; % exp(Vw)];
        M = size(embedding, 1);
        output1 = size(embedding);
        output2(:,1) = (0:M-1)';
        output2(:,2:size(embedding,2)+1) = embedding;
        outputname = '4406_exp_svd_U_V.emb';
        dlmwrite( outputname, output1, 'delimiter',' ');
        dlmwrite( outputname, output2,'-append', 'delimiter',' ');
     
