

addpath('/Users/sihengc/Documents/Research/Project/Graph_Signal_Processing/Toolbox/Social_Science/embedding/dw_scoring/benchmark/task1_dw_converge/data/kaggle/')
files = dir('/Users/sihengc/Documents/Research/Project/Graph_Signal_Processing/Toolbox/Social_Science/embedding/dw_scoring/benchmark/task1_dw_converge/data/kaggle/*.mat');

Walk = {'finite', 'infinite'}
Poly_power =  [1  0.1  0.01  0  -0.01  -0.1  -1];

% Wrap = {'exp', 'tan'};
% Emb = {'svd', 'cross_entropy'}
option.dimension = 10;
option.step_number = 7;
option.flyout = 0.85;
option.wrap = 'poly';
option.emb = 'svd';   %   svd  cross_entropy


for i_file = 1:length(files)
    
    load(files(1).name)
   
    A = network;
    filename = strsplit( files(i_file).name, '.');
    
    
    for i_walk = 1:length(Walk)  

            option.mode = Walk{i_walk};
            Pi = runRW(A, option);
            
            for i_power = 1:length(Poly_power)
                
                    option.poly_power = Poly_power(i_power);
                    [Uw Vw Y]= obtainEmb(Pi, option);


                    embedding = [Uw Vw]; % exp(Vw)];
                    M = size(embedding, 1);
                    output1 = size(embedding);
                    output2(:,1) = (0:M-1)';
                    output2(:,2:size(embedding,2)+1) = embedding;
                    outputname = [filename{1}  '_' option.mode  '_'   option.wrap  '_'  num2str(i_power)  '.emb'];
                    dlmwrite( outputname, output1, 'delimiter',' ');
                    dlmwrite( outputname, output2,'-append', 'delimiter',' ');
                    
            end
            
    end

end

