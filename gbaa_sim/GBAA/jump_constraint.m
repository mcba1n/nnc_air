function [At, ft, I_sub] = jump_constraint(A, f, J)
%Remove nodes below threshold J

% remove nodes that violate the minimum jump constraint
for i = 1:size(A,1)
   for j = 1:size(A,2)
       if A(i,j)==1 && abs(f(i)-f(j)) < J
           A(i,j)=0; % remove node
       end
   end
end

% choose the largest fully-connected component
g = digraph(A);
[bins,binsizes] = conncomp(g);
[~,K_max] = max(binsizes);

% create the new sub-graph
I_sub = find(bins==K_max);
At = A(I_sub,I_sub);
ft = f(I_sub);

end

