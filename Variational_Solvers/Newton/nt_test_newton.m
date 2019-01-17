clearvars; close all; clc; 
addpath('../../')

% Written by Mauricio Flores, last edited December 2018.
% For any questions, email mauricio.a.flores.math@gmail.com
% For details, you can also see our paper:
% 'Algorithms for Lp-based semi-supervised learning on graphs'

% Setup Problem Parameters:
p = 5; n = 1e4; k_neigh = 10; tol = 1e-8; dim = 5;

% Generate Random Data & Compute Weights:
[X, Y, g, m] = data_dim_d(n, dim);
[knn, wnn, k_neigh] = compute_knn_wnn([X; Y], n, m, k_neigh);

% Define initial guesses for 'u'
u0 = mean(g)*ones(n+m,1) + 0.01*randn(n+m,1); nt_iter = 20;



%% Homotopy approach:
% The simplest approach is to pick some values of 'p'
% which gradually approach the value of 'p' you want to
% solve for. You can take increasing step-sizes in 'p',
% for example, if the goal is 50, you can do 
% p = 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50.
% Always start from p = 2. Here we provide a more 
% automatic way to do this.
u = nt_solve_newton(g, n, m, 2, k_neigh, u0, ...
                       knn, wnn, nt_iter, tol);
                 
% 
p_local = 2; factor = 1.5; 
% You can try bigger or smaller 'factor'.
% Bigger 'factor' will step through 'p' more
% quickly, but each problem (for a fixed 'p')
% might take more iterations if the step in 'p'
% gets proportionally too big. Try 1.1 < factor < 2.5
while 1
    
    p_local = p_local * factor;
    
    if p_local >= p 
        p_local = p; 
        
        u = nt_solve_newton(g, n, m, p_local, k_neigh, u, ...
                                knn, wnn, nt_iter, tol);
        break;
    
    else
        u = nt_solve_newton(g, n, m, p_local, k_neigh, u, ...
                                knn, wnn, nt_iter, tol);
    end
    
end   


%u = nt_solve_newton(g, n, m, p, k_neigh, u, knn, wnn, nt_iter, tol);