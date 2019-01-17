function u = newtonlike_solve(g, n, m, k_neigh, ...
             p, knn, wnn, dx, u, bt_iter, tol, iterative)

fprintf('\nWorking on p = %.2f\n', p)
         
if nargin < 13; iterative = 0; end % By default, we solve using backslash
                                   
% Attach Boundary Condition:                    
u = [u; g];

% Solve iteratively:
for iter = 1 : bt_iter
    
    %% Compute Error: 
    grad_u = wnn .* (u(knn) - u(1:n)); 
    grad_p = game_compute_grad_p(grad_u, dx, p);
    
    err = max(abs(grad_p));
    fprintf('Iter = %d, Res %.3e \n', iter, err)

    if err < tol; break; end % Break out of loop if done
    
    % Solver:
    u = newtonlike_matrix(grad_u, g, knn, wnn, dx, p, n, m, ...
                   k_neigh, iterative, u(1:n), err);
end

u = u(1:n);

end
