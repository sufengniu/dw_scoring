function [Pinv P] = mergeTransition( mu, PG, P_Ai, alpha )

N = size(PG,1);
P = (1- sum(mu)).*PG;
for i = 1:length(P_Ai)
    P = P + mu(i).*P_Ai{i};
end
Pinv = inv( eye(N) - alpha.*P);
% Y = log( ( Pinv - eye(N) )./alpha + eps^2);