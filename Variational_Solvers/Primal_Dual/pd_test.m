clearvars; close all; clc; 
addpath('../../')

% Written by Mauricio Flores, last edited December 2018.
% For any questions, email mauricio.a.flores.math@gmail.com
% For details, you can also see our paper:
% 'Algorithms for Lp-based semi-supervised learning on graphs'

% This method works, at least for moderate 'p' values, but speed
% depends very heavily on a good choice of gamma (otherwise the
% algorithm takes thousands of iterations to gain any accuracy.
% We recommend Newton's method with homotopy instead, but we 
% provide this code as a starting point in case there is interest
% in exploring other approaches to improve performance here.

% BTW, we can also improve performance by replacing the bisection
% solver with the gradient solver (for the dual step), but in that
% case, we can only converge up to a point, and after that, the 
% inexact gradient step prevents further progress.

% Setup Problem Parameters:
p = 3; q = p / (p - 1);
n = 1e4; k_neigh = 10; rho = 1.000;

% Generate Random Data & Compute Weights:
dim = 10; [X, Y, g, m] = data_dim_d(n, dim);

% Compute step - tuning:
gamma = n^(1/dim) * 10^(2*p-5);
% WARNING: this formula is just an ad-hoc recommendation.
% Choosing a good gamma, for a specific problem, is probably necessary.

% Compute Weights:
[knn, wnn, k_neigh, sigma] = compute_knn_wnn([X;Y], n, m, k_neigh);

% Specify tolerance (scaled by length-scale)
tol_scaling = n * sigma^(dim+p-1);
tol = 1e-12 * tol_scaling;
                                     
% Identify indexes that need updating for v(x,y):
[idx, w_conj] = pd_idx_to_solve(knn, wnn, q);

% Define initial guesses for 'u', 'ubar' and 'v':
u = [mean(g)*ones(n,1);g]; ubar = [mean(g)*ones(n,1);g];
v = zeros(n+m, k_neigh); % Skew-symmetric initial guess

% Primal Dual Solve:
dual_solver = 2; iterations = 1e3;
u = pd_solve(knn, wnn, w_conj, u, ubar, v, ...
    idx, g, p, q, n, gamma, iterations, rho, tol, dual_solver);
