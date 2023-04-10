function [z] = eexp(x)

if isinf(x)
    z = 0;
else
    z = exp(x);
end

end

