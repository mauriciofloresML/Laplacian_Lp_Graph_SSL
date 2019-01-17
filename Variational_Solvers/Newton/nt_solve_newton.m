function [u, res, iter] = nt_solve_newton(g, n, m, p, k_neigh, u, ...
                                         knn, wnn, nt_iter, tol)

fprintf('\nWorking on p = %.2f\n', p)

% Newton coeffs (q1 = 1, q2 = 0 => IRLS)
q1 = 1/(p-1); q2 = (p-2)/(p-1); 

for iter = 1 : nt_iter
    
    u(n+1:n+m) = g; % Attach boundary condition:
    
    % Compute Error, and stop if tolerance is achieved
    grad_p = sum(wnn .* (u - u(knn)).* abs(u - u(knn)).^(p-2), 2);
    res = max(abs(grad_p(1:n)));
    
    % Print Result & Break if Tolerance achieved:
    fprintf('Iter %d = , Res %.3e \n', iter, res)
    if res < tol; break; end
    
    % If tolerance not achieved yet, then compute next:
    [Lap_Bg] = nt_invert_Lap(wnn, knn, u, g, n, m, k_neigh, p, 1, res);
          
    u = q1 * Lap_Bg + q2 * u(1:n);
    
end
    
    u(n+1:n+m) = g; % In case tolerance isn't reached, at least impose B.C.

end
