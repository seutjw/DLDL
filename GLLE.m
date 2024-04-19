function [pred,num,P] = GLLE(logicalLabel, features,lambda2,lambda3,lambda4,labels)
[fea,dim] = size(features);
[~,l] = size(logicalLabel);
global   R;
global   trainFeature;
global   trainLabel;
global   G;
global   para;
global   H;
global   F;

global   l1;
global   l2;
global   l3;
global   l4;
global   l5;



l2=lambda2;
l3=lambda3;
l4=lambda4;
labels=labels+eps;
% K-NN similarity matrix A
T=5;
Idx = knnsearch(features,features,'K',20);
GraphConnect = zeros(size(features,1),size(features,1));
for i = 1:size(features,1)
    GraphConnect(i,Idx(i,:)) = 1;
end
GraphConnect = GraphConnect + GraphConnect';
GraphConnect(GraphConnect > 0) = 1;
sigma = 20;
A =  exp(-(L2_distance(features', features').^2) / (2 * sigma ^ 2));
A = A .* GraphConnect;
A = A - diag(diag(A));
A_hat = diag(sum(A,2));
G = A_hat - A;
GF=G;
G=G*G';
%type of kernel function ('lin', 'poly', 'rbf', 'sam')
ker  = 'rbf';
%parameter of kernel function
par  = 1*mean(pdist(features));
% build the kernel matrix on the labeled samples (N x N)
H = kernelmatrix(ker, par, features, features);
UnitMatrix = ones(size(features,1),1);
trainFeature = [features];
trainLabel = logicalLabel;
% para = lambda;
item=rand(size(trainFeature,2),size(trainLabel,2));
AA=ones(1,size(trainLabel,2));
lb=zeros(1,size(trainLabel,2));
options_qp=optimoptions('quadprog','Display','off');
% F=rand(size(trainLabel));
% F=trainLabel;
% m=mean(F,2);
% me=repmat(m,1,size(F,2));
% F=F.*me;

sum_label = sum(logicalLabel');
min_num = min(sum_label);

c=size(logicalLabel,2);
[num, dim] = size(features);
Q = zeros(c,dim);
for ii = 1 : 1 : c
    for kkk = min_num : 1 : c
        tt = 1;
        for i = 1 : 1 : num
            if((sum_label(i) == kkk) && (logicalLabel(i,ii) == 1))
                Q(ii,:) = Q(ii,:) + features(i,:);
                tt = tt+1;
            end
        end
        Q(ii,:) = Q(ii,:)/tt;
        sum_Q = sum(Q(ii,:));
        if sum_Q > 0
            break;
        end
    end
end

Length = sqrt(sum(Q.^2, 2));
Length(Length <= 0) = 1e-8;          % avoid division by zero problem for unlabeled rows
rho = 1 ./ Length;
Q_1 = diag(sparse(rho)) * Q;

Length_1 = sqrt(sum(features.^2, 2));
Length_1(Length_1 <= 0) = 1e-8;
rho_1 = 1 ./ Length_1;
features_1 = diag(sparse(rho_1)) * features;

S_1 = features_1 * (Q_1');
P = S_1 + logicalLabel;
P = (softmax(P'))';

F=P;
W=eye(size(features,2),size(logicalLabel,2));
disp('Initializing... (total 10)');
lambda3_2=3000;
for t=1:10
    disp(t);
    pred=logicalLabel;
    for j=1:size(F,1)
        vecF=F(j,:);
        vecY=logicalLabel(j,:);
        F_prog=zeros(1,size(F,2));
        for i=1:size(F,1)
            if(i==j)
                continue;
            end
            s=GF(i,j)+GF(j,i);
            F_prog=F_prog+F(i,:)*s-2/lambda2*pred(i,:);
        end
        G_prog=eye(size(F,2))*(GF(j,j)+(1+lambda3_2)/lambda2);
        [optF]=quadprog(G_prog,F_prog,[],[],AA,1,lb,vecY,vecF,options_qp);
        F(j,:)=optF;
        
        
    end
    trainLabel=F;
end
% load('F.mat');
num=F;
num=num+eps;
labels=labels+eps;
trainDistribution=F;
disp('training W+D... (total 5)');
for t=1:T
    disp(t);
    
    fai1=zeros(size(F));
    tao=0.001;
    A=fai1;
    save('dat.mat','trainDistribution','trainFeature','W','lambda2','lambda3','lambda4','G','fai1','tao','A');
    optim=optimset('Display','on','GoalsExactAchieve','0','MaxIter',50);
    [W,fval]=fminlbfgs(@bfgsProcessW,W,optim);
    while(tao<0.002)
        fprintf("tao=%f\n",tao);
        optim=optimset('Display','on','GoalsExactAchieve','0','MaxIter',50);
        save('dat.mat','F','trainFeature','W','lambda2','lambda3','lambda4','G','fai1','tao','A');
        %[F,fval]=fminlbfgs(@bfgsProcessF,F,optim);
        it=0;
        pref=0;
        while(it<50)
            [fval,gradient]=bfgsProcessF(F);
            F=F-0.00001*gradient;
            it=it+1;
            if(abs(pref-fval)<1e-3)
                break;
            end
            pref=fval;
        end
        Gprog=eye(size(F,2),size(F,2))*(1+1/tao);
        for j=1:size(F,1)
            Hprog=zeros(1,size(F,2));
            vecY=logicalLabel(j,:);
            vecA=A(j,:);
            for i=1:size(F,1)
                if(i==j)
                    continue;
                end
                Hprog=Hprog+2*A(i,:)-2*F(i,:)+2/tao*fai1(i,:);
            end
            [optA]=quadprog(Gprog,Hprog,[],[],AA,1,lb,vecY,vecA,options_qp);
            A(j,:)=optA;
        end
        fai1=fai1+tao*(A-F);
        tao=1.2*tao;
        trainDistribution=F;
        num=A;
    end
num=num+eps;
pred=W;
P=F;

save('W.mat','W');
end
