clearvars; close all; clc;
addpath('../../')

% Setup problem:
n = 1e4; p = 5; k_neigh = 10; tol = 1e-5;
[X, Y, g, m] = data_dim_d(n, 10);

% Compute weights:
[knn, wnn, k_neigh] = compute_knn_wnn([X;Y], n, m, k_neigh);
W = sparse((1:n+m)'*ones(1, k_neigh), knn, wnn);
wnn = wnn(1:n,:); knn = knn(1:n, :); dx = sum(wnn, 2);

%Solve p Laplacian learning problem
u = semi_solve_fast(W, g, p, knn, wnn, dx, n, tol);