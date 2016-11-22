
filename = 'Flickr';
load(['../data/' filename '.mat'])


A = network;
option.mode = 'finite';
option.step_number = 7;
[Y Pi Uw] = runRW(A, option);

embedding = Uw;
M = size(embedding, 1);
output1 = size(embedding);
output2(:,1) = (0:M-1)';
output2(:,2:size(embedding,2)+1) = embedding;
filename = [filename '_finite.embeddings'];
dlmwrite( filename, output1, 'delimiter',' ');
dlmwrite( filename, output2,'-append', 'delimiter',' ');


option.mode = 'infinite';
option.flyout = 0.85;
[Y Pi Uw] = runRW(A, option);

embedding = Uw;
M = size(embedding, 1);
output1 = size(embedding);
output2(:,1) = (0:M-1)';
output2(:,2:size(embedding,2)+1) = embedding;
filename = [filename '_infinite.embeddings'];
dlmwrite( filename, output1, 'delimiter',' ');
dlmwrite( filename, output2,'-append', 'delimiter',' ');

