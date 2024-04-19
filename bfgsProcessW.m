function [target,gradient] = bfgsProcessW(weights)
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
modProb = exp(trainFeature * weights);  % size_sam * size_Y
sumProb = sum(modProb, 2);
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
%disp(modProb);
% Target function.
target = -sum(sum(trainDistribution.*log(modProb)))+lambda4*norm(weights,'fro')^2;

% The gradient.
W=2*lambda4*weights;
gradient =zeros(size(weights));
for k=1:size(gradient,1)
    for j=1:size(gradient,2)
        sum1=0;sum2=0;
        for i=1:size(trainFeature,1)
            sum1=sum1+modProb(i,j)*trainFeature(i,k);
            sum2=sum2+trainDistribution(i,j)*trainFeature(i,k);
        end
        gradient(k,j)=sum1-sum2+W(k,j);
    end
end
end
