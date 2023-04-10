function A_noisy = gbaa_noisy_adj(T_est, A)

A_noisy = zeros(size(A,1),size(A,2));
for i = 1:size(A,1)
    for j = 1:size(A,2)
        if A(i,j) == 1
            A_noisy(i,j) = 2^T_est(i,j);
        end
    end
end

end

