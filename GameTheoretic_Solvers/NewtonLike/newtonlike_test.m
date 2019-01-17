clearvars; close all; clc;
addpath('../../')

n = 1e4; k_neigh = 10; p = 5; dim = 5;

[X, Y, g, m] = data_dim_d(n, dim);
[knn, wnn, k_neigh] = compute_knn_wnn([X;Y], n, m, k_neigh); 

% These computations are needed for consistency.
wnn = wnn(1:n,:); knn = knn(1:n,:); dx = sum(wnn, 2);

%% Compute Solution with Beta Algorithm
bt_iter = 100; u = zeros(n,1);

%% Homotopy approach:
% The simplest approach is to pick some values of 'p'
% which gradually approach the value of 'p' you want to
% solve for. You can take increasing step-sizes in 'p',
% for example, if the goal is 50, you can do 
% p = 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50.
% Always start from p = 2. Here we provide a more 
% automatic way to do this.
u = newtonlike_solve(g, n, m, k_neigh, 2, knn, wnn, ...
                     dx, u, bt_iter, 1e-12, 1);
                 
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
        
        u = newtonlike_solve(g, n, m, k_neigh, p_local, ...
                    knn, wnn, dx, u, bt_iter, 1e-12, 1);
        break;
    
    else
        u = newtonlike_solve(g, n, m, k_neigh, p_local, ...
                    knn, wnn, dx, u, bt_iter, 1e-12, 1);
    end
    
end   