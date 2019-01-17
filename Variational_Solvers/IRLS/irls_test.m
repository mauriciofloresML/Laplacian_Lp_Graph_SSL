clearvars; close all; clc; 
addpath('../../')   

% Written by Mauricio Flores, last edited December 2018.
% For any questions, email mauricio.a.flores.math@gmail.com
% For details, you can also see our paper:
% 'Algorithms for Lp-based semi-supervised learning on graphs'

% We do not recommend using this algorithm, because it only
% converges reliably when p < 3, and even in those cases, it
% takes a lot more iterations than Newton's method does.

% Anyway, if interested, you can run a simple IRLS experiment
% with this code. If you wish to see a failure, pick p > 3.

% Setup Problem Parameters:
p = 2.5; n = 1e4; k_neigh = 10; iterations = 100; tol = 1e-12;

% Generate Random Data & Compute Weights:
[X, Y, g, m] = data_dim_2(n);
[knn, wnn, k_neigh] = compute_knn_wnn([X; Y], n, m, k_neigh);                             

% Define initial guesses for 'u'
u = mean(g)*ones(n+m,1) + 0.01*randn(n+m,1); 

% Solve Primal Dual Problem Iteratively:
u = irls_solve(g, n, m, p, k_neigh, u, knn, wnn, iterations, tol);