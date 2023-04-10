function [log_post,F] = F_hmm(y, m, P, f, sigma, K, s_0)

Nstates = size(P,1);
F = Inf*ones(Nstates,m);
log_P =  arrayfun(@eln, P);

%% initialisation
for s = 1:Nstates
    log_gamma = log_P(s_0, s) + eln(normpdf(y(1), f(s), sigma/sqrt(K(1))));
    F(s,1) = elnsum(F(s,1), log_gamma);
end

%% recursion
for ell = 2:m 
    for s = 1:Nstates
        prev_state = find(P(:,s) > 0);
        for i = 1:length(prev_state) 
            log_gamma = log_P(prev_state(i), s) + eln(normpdf(y(ell), f(s), sigma/sqrt(K(ell))));
            F(s,ell) = elnsum(F(s,ell), elnproduct(log_gamma, F(prev_state(i),ell-1)));  
        end
    end
     %fprintf('F: (%d / %d)\n', ell, m);
end

%% termination
log_post = Inf;
for s = 1:Nstates
    log_post = elnsum(log_post, F(s,m));
end

end

