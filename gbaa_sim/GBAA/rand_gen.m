function X=rand_gen(x,pmf,N)
% X=rand_gen(x,pmf,N)
% Input:
% x   : set of the all possible values that the desired random signal can
%       assume
% pmf : vector that cointains the probability of each possible 
%       value of x
% output:
% X   : random signal whit the desired pmf
%
% Example: 
% pmf=[1/3 1/3 1/3]
% x=[1 2 3];
% N=100;
% X=rand_gen(x,pmf,N);

a = [0; cumsum((pmf(:)))]*(1./rand(1,N));
b = a > ones(length(pmf)+1,N);
[c,index] = max(b);
x = x(:);
X = x(index-1);
