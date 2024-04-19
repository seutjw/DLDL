function [E]=oneerror(gd,pred)
    E=0;
    for i=1:size(gd,1)
        for j=1:size(gd,2)
            if(gd(i,j)>0.01) 
                gd(i,j)=1;
            else 
                gd(i,j)=0;
            end
            if(pred(i,j)>0.01) 
                pred(i,j)=1;
            else 
                pred(i,j)=0;
            end
            if(gd(i,j)==pred(i,j))
                E=E+1;
            end
        end
    end
    E=E/(size(gd,1)*size(gd,2));
end