function [u, err] = gd_solve(g, n, p, knn, wnn, dx, u, gd_iter, tol)

u = [u; g];

% This bound could be changed, but no need to do it.
alpha = p / (2*p - 3);

err = 1e6 * ones(gd_iter, 1);
for iter = 1 : gd_iter
     
    grad_u = wnn .* (u(knn) - u(1:n));
    grad_p = game_compute_grad_p(grad_u, dx, p);
 
    % Gradient Descent Step:
    u = [u(1:n) + alpha * grad_p; g];
    
    %% Compute Error:
    err(iter) = max(abs(grad_p));
    if err(iter) < tol; break; end
    
    if mod(iter, 50) == 1
        fprintf('Error = %.3e, at Iter = %d\n', max(abs(grad_p)), iter)
    end

end

end
