function K = kernelmatrix(ker, parameter, testX, trainX)

switch ker
    case 'lin'
        if exist('trainX','var')
            K = testX * trainX' + parameter;
        else
            K = testX * testX' + parameter;
        end

    case 'poly'
        if exist('trainX','var')
            K = (testX * trainX' + 1).^parameter;
        else
            K = (testX * testX' + 1).^parameter;
        end
        
    %To speed up the computation of the RBF kernel matrix, 
    %we exploit a decomposition of the Euclidean distance (norm).
    case 'rbf'  
        n1sq = sum(testX'.^2,1); %compute x^2
        n1 = size(testX',2);
        if isempty(trainX);
            %||x-y||^2 = x^2 + y^2 - 2*x'*y 
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*testX*testX';
        else
            n2sq = sum(trainX'.^2,1);
            n2 = size(trainX',2);
            %||x-y||^2 = x^2 + y^2 - 2*x'*y 
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*testX*trainX'; 
        end;
        K = exp(-D/(2*parameter^2));

    case 'sam'
        if exist('trainX','var');
            D = testX*trainX';
        else
            D = testX*testX';
        end
        K = exp(-acos(D).^2/(2*parameter^2));

    otherwise
        error(['Unsupported kernel ' ker])
end
