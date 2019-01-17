function u = pd_solve(knn, wnn, w_conj, ...
    u, ubar, v, idx, g, p, q, n, gamma, pd_iter, rho, tol, dual_method)

% Compute step sizes:
k_neigh = size(knn,2); 
r1 = (rho / gamma) / sqrt(4*k_neigh); 
r2 = (rho * gamma) / sqrt(4*k_neigh); 

grad_p = sum(wnn .* (u - u(knn)).* abs(u - u(knn)).^(p-2), 2);
max(abs(grad_p(1:n)))

for i = 1 : pd_iter

    u0 = u;     % We save old iteration for over-relaxation
       
    % Dual Step:
    v = pd_dual_step(knn, w_conj, v, ubar, idx, r1, q, dual_method);             
    
    % Primal Step:
    u = u0 - r2 * (2*sum(v,2));
    u(n+1:end) = g;
    
    % Over-relaxation
    ubar = 2*u - u0;
    
    % Compute res & chg:
    grad_p = sum(wnn .* (u - u(knn)).* abs(u - u(knn)).^(p-2), 2);
    res = max(abs(grad_p(1:n)));
    
    if res < tol; break; end
    
    if mod(i, 10) == 0 % Print error once in a while
        fprintf('iter = %d, res = %.3e\n', i, res)
    end
    
end

end