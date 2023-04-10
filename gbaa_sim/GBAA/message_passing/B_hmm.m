function [B] = B_hmm(y, m, P, f, sigma, K)

Nstates = size(P,1);
B = Inf*ones(Nstates,m);
log_P =  arrayfun(@eln, P);

%% initialisation
for s = 1:Nstates
    B(s,m) = eln(1);
end

%% recursion
for ell = m-1:-1:1  
    for s = 1:Nstates   
        next_state = find(P(s,:) > 0);

        for i = 1:length(next_state)  
            log_gamma = log_P(s, next_state(i)) + eln(normpdf(y(ell+1), f(next_state(i)), sigma/sqrt(K(ell+1))));
            B(s,ell) = elnsum(B(s,ell), elnproduct(log_gamma, B(next_state(i),ell+1)));
        end

    end
     %fprintf('B: (%d / %d)\n', ell, m);
end

end

