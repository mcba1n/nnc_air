function [z] = elnproduct(x,y)

if isinf(x) || isinf(y)
    z = Inf;
else
    z = x+y;
end

end

