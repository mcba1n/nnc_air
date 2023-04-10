function [P,mu,H] = max_source_ent(A)

% the maximum entropy transition probabilities
[V,D,~] = eig(A);
[dom_eig, i_max] = max(abs(diag(D)));
wts = V(:,i_max);
H = log2(dom_eig);

P = zeros(size(A,1),size(A,2));
for i = 1:size(A,1)
    for j = 1:size(A,2)
        if A(i,j) == 0
            continue;
        end
        P(i,j) = (wts(j,1)/wts(i,1))*(A(i,j)/dom_eig);
    end
end

% stationary distribution
mu = null(eye(size(P))-P.');
mu = mu./sum(mu);

end

