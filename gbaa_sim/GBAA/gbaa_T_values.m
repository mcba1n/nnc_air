function T_est = gbaa_T_values(log_post, F, B, P, mu, f, sigma, K, y)

Nstates = size(F,1);
m = size(F,2);
T_est = zeros(Nstates,Nstates);
log_P =  arrayfun(@eln, P);

for ell = 2:m
    for s1 = 1:Nstates
        for s2 = find(P(s1,:)>0)
            % psi
            log_gamma = log_P(s1, s2) + eln(normpdf(y(ell), f(s2), sigma/sqrt(K(ell))));
            log_psi1 = F(s1,ell-1) + log_gamma + B(s2,ell) - log_post;
            psi1 = eexp(log_psi1);
            
            % joint psi
            log_psi2 = elnproduct(F(s1,ell-1), B(s1,ell-1)) - log_post;
            psi2 = eexp(log_psi2);
            
            % T-value
            num = psi1/(mu(s1)*P(s1,s2))*log2(psi1);
            denom = psi2/mu(s1)*log2(psi2);
            
            if psi1 == 0
                num = 0;
            end
            
            if psi2 == 0
                denom = 0;
            end
            
            T_val = num - denom;
            T_est(s1,s2) = T_est(s1,s2) + T_val/m;
        end
    end
end

end

