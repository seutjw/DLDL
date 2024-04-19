function [target,gradient] = targetFunc(weights)

% Load the data set.
global   trainFeature;
global   trainLabel;
global   G;
global   para;
%load dt.mat;
[size_sam,size_X]=size(trainFeature);

modProb =  trainFeature * weights;  % size_sam * size_Y

L = sum(sum((modProb - trainLabel).^2)); 
R =  trace(modProb'*G*modProb);
gradL = 2*trainFeature' * (modProb - trainLabel);
gradR =  trainFeature'*G*trainFeature*weights + ( trainFeature'*G*trainFeature)'*weights;

target =( L + para * R );
gradient = (gradL +  para * gradR);
end


