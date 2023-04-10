function [z] = elnsum(x,y)

if isinf(x) || isinf(y)
    if isinf(x)
        z = y;
    else
        z = x;
    end
else    
    if x > y
        z = x + log1p(exp(y - x));
    else
        z = y + log1p(exp(x - y));
    end
end
    

end

