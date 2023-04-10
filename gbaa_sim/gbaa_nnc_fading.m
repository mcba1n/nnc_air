clear all; close all; clc;
rng(1,'twister');
addpath('GBAA/');
addpath('GBAA/message_passing/');
addpath('GBAA/log_semiring/');

% Load the state-space
load('scrappie_graph.mat');
J = 0;
[A, f] = jump_constraint(A, f, J);
Nstates = length(f);

% IUD Markov source
[P_iud,mu_iud,H] = max_source_ent(A);

% Simulation parameters
sigma_vals = 0.05:0.05:1.00;
max_iters = 20;
eps = 0.001;
m = 10^5;
rates_vec = zeros(length(sigma_vals),max_iters);
optimised_rates_vec = zeros(1,length(sigma_vals));
optimised_P = cell(1,length(sigma_vals));
P_K = [0.5,0.5]; %[0,0,0,0.5,0,0,0,0.5]; 

% Run simulation
for idx = 1:length(sigma_vals)
    sigma = sigma_vals(idx);
    fprintf('Starting optimisation for sigma=%.2f:\n', sigma);
    P = P_iud;
    mu = mu_iud;
    for iter = 1:max_iters
        tic
        % Generate random observations
        s_0 = 1;
        mc = dtmc(P);
        x0 = zeros(1,Nstates);
        x0(s_0) = 1;
        S = simulate(mc,m,'X0',x0);
        K = [];
        y = [];
        for i = 1:m
            k_i = rand_gen(1:length(P_K),P_K,1);
            y = [y, normrnd(f(S(i+1)), sigma/sqrt(k_i))];
            K = [K, k_i];
        end

        % Forward and backward probabilities
        [log_post,F_log] = F_hmm(y, m, P, f, sigma, K, s_0);
        fprintf('F done.\n');
        B_log = B_hmm(y, m, P, f, sigma, K);
        fprintf('B done.\n');

        % Compute information rate
        T_est = gbaa_T_values(log_post, F_log, B_log, P, mu, f, sigma, K, y);
        I = gbaa_air(T_est, P, mu);
        rates_vec(idx, iter) = I;
        fprintf('%.4f\n', I);

        % Update Markov source
        A_noisy = gbaa_noisy_adj(T_est, A);
        [P, mu, ~] = max_source_ent(A_noisy);

        % Check early stopping condition
        if iter > 1
            if abs(rates_vec(idx, iter)-rates_vec(idx, iter-1)) < eps
                optimised_P{1,idx} = P;
                break;
            end
        end
        toc
    end
    optimised_rates_vec(idx) = I;
    fprintf('Optimised rate: %.4f\n', I);
end

%% Plot the optimised rates with lower bound
semilogx(snr_vals, max(rates_vec,[],2), '-om'), hold on, grid on;
semilogx(snr_vals, rates_vec(:,1), '-ok'), hold on, grid on;
legend('Optimised', 'IUD', 'location', 'SouthEast')
xlabel('SNR');
ylabel('Information rate (bits/symbol)');
title('Information rates');

%% Save data
save('optimised_P.mat', 'optimised_P', 'sigma_vals', 'rates_vec', 'optimised_rates_vec');