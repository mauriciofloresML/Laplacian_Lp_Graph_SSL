% This solver, it only converges reliably when p < 3.
% For better performance, use Newton's method with homotopy.

function u = irls_solve(g, n, m, p, k_neigh, u, knn, wnn, irls_iter, tol)

for iter = 1 : irls_iter
    
    u(n+1:n+m) = g; % Attach boundary condition
    
    % Compute Error, and stop if tolerance is achieved
    grad_p = sum(wnn .* (u - u(knn)).* abs(u - u(knn)).^(p-2), 2);
    res = max(abs(grad_p(1:n)));
    
    % Print Result & Break if Tolerance achieved:
    fprintf('Iter %d = , Res %.3e \n', iter, res)
    if res < tol; break; end
    
    % Compute Next Iterate & attach B.C.
    u = [irls_invert_Lap(wnn, knn, u, g, n, m, k_neigh, p); g]; 
    
end

end
