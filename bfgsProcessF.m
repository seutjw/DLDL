function [target,gradient] = bfgsProcessF(trainDistribution)
%BFGSPROCESS	Provide the target function and the gradient.
%
%	Description
%   [TARGET,GRADIENT] = BFGSPROCESS(WEIGHTS) provides the target function and the gradient.
%   They will be used in the optimization of BFGSLLD -- BFGSLLDTRAIN.
%   
%   Inputs,
%       WEIGHTS: the weights which will be optimized in BFGSLLDTRAIN
%   Outputs,
%       TARGET:  the target function which will be used in BFGSLLDTRAIN
%       GRADIENT: the gradient which will be used in BFGSLLDTRAIN
% 
%	See also
%	BFGSLLDTRAIN, LLDPREDICT, FMINLBFGS 
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

% Load the data set.
load('dat.mat');
% modProb = exp(trainFeature * W);  % size_sam * size_Y
% sumProb = sum(modProb, 2);
% modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
% 
% % Target function.
pred=lldPredict(W,trainFeature);
dp=abs(trainDistribution).*log(abs(trainDistribution./pred));

for i=1:size(dp,1)
    for j=1:size(dp,2)
        if(isinf(-dp(i,j))) 
            dp(i,j)=0;
        end
    end
end

target = sum(sum(dp))+lambda3*norm(trainDistribution,'fro')^2+lambda2*trace(trainDistribution'*G*trainDistribution)+trace(fai1'*(A-trainDistribution))+tao/2*norm(A-trainDistribution,'fro')^2;

gradient=ones(size(trainDistribution))+log(abs(trainDistribution))-log(abs(pred));

for i=1:size(gradient,1)
    for j=1:size(gradient,2)
        if(isinf(-gradient(i,j))) 
            gradient(i,j)=0;
        end
    end
end

% The gradient.
gradient = gradient+2*lambda3*trainDistribution+lambda2*((G+G')*trainDistribution)-fai1-tao*(A-trainDistribution);
%disp(gradient);
% pause;
end
