clearvars; close all; clc;
addpath('../../')

% 
n = 1e4; p = 3; k_neigh = 10; dim = 3; 

% 
[X, Y, g, m] = data_dim_d(n, dim); 
[knn, wnn, k_neigh, sigma] = compute_knn_wnn([X;Y], n, m, k_neigh); 

% These computations are needed for consistency.
wnn = wnn(1:n,:); knn = knn(1:n,:); dx = sum(wnn, 2);

%% Compute Solution with Bisection Solver
u0 = 0.5*ones(n,1); iterations = 1e5;
u = gd_solve(g, n, p, knn, wnn, dx, u0, iterations, 1e-11);