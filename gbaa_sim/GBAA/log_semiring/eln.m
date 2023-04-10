function [z] = eln(x)

if x == 0
    z = Inf;
else
    z = log(x);
end

end

