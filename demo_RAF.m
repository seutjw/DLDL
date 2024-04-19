clc
clear
addpath(genpath(pwd));
addpath './datasets';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Selecting the dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset={'data_train_RAF'};
T=strcat(dataset(1),'.mat');
load(T{1,1});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initializating training and testing matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_tr=double(X_tr);
Dr_tr=double(Dr_tr);
cut1=floor(size(X_tr,1)*0.6);
cut2=floor(size(X_tr,1)*0.8);
cut3=floor(size(X_tr,1)*1);
X_cross=X_tr(cut1+1:cut2,:);
X_test=X_tr(cut2+1:cut3,:);
Dr_test=Dr_tr(cut2+1:cut3,:);
X_tr=X_tr(1:cut1,:);
Dr_tr=Dr_tr(1:cut1,:);
features=X_tr;
labels=Dr_tr;
logicalLabel=zeros(size(labels));
for i=1:size(Dr_tr,1)
    for j=1:size(Dr_tr,2)
        if(Dr_tr(i,j)>0.01) 
            logicalLabel(i,j)=1;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Appointing parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda2=0.1;%alpha
lambda3=0.001; %beta
lambda4=0.01;%gamma

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[W, num,P] = GLLE(logicalLabel, features ,lambda2,lambda3,lambda4,Dr_tr);
num=num+eps;
ker  = 'rbf'; 
par1  = 1*mean(pdist(X_tr)); 
par2  = (1*mean(pdist(X_tr))+1*mean(pdist(X_test)))/2; 
UnitMatrix1 = ones(size(X_tr,1),1);
UnitMatrix2 = ones(size(X_test,1),1);
H1 = kernelmatrix(ker, par1,   X_tr, X_tr);
H2 = kernelmatrix(ker, par2, X_test, X_tr);
distribution = (softmax((X_tr*W)'))'; 
save('distribution.mat','distribution');

LDL=lldPredict(W,X_test);
Dr_test=Dr_test+eps*ones(size(Dr_test));

for i=1:size(LDL,1)
    for j=1:size(LDL,2)
        if(isnan(LDL(i,j)))
            LDL(i,j)=1;
        end
    end
end
LDL=LDL+eps;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Organizing the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ResultD = zeros(4,1);
ResultD(1,1) = chebyshev(Dr_test,LDL);
ResultD(2,1) = clark(Dr_test,LDL);
ResultD(3,1) = oneerror(Dr_test,LDL);
ResultD(4,1) = intersection(Dr_test,LDL);



save('ResultD.mat','ResultD');
disp('Predictive results:');
disp(ResultD);
disp('-----------------------');

labels=labels+eps;
ResultP = zeros(4,1);
ResultP(1,1) = chebyshev(labels,num);
ResultP(2,1) = clark(labels,num);
ResultP(3,1) = oneerror(labels,num);
ResultP(4,1) = intersection(labels,num);

disp('Recovery results:');
disp(ResultP);
save('ResultLE.mat','ResultP');
