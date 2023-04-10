function I = gbaa_air(T, P, mu)

Nstates = size(P,1);
I = 0;
for s1 = 1:Nstates
    for s2 = 1:Nstates
        if P(s1,s2)==0
            continue;
        end
        I = I + mu(s1)*P(s1,s2)*(log2(1/P(s1,s2)) + T(s1,s2));
    end
end

end

