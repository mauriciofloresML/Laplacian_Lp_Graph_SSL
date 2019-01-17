function u = semi_solve_fast(W, g, p, knn, wnn, dx, n, tol)

if p == Inf; alpha = 0; delta = 1;
else; alpha = 1/p; delta = 1-2/p; end

%Right hand side of linear system
b = W*[zeros(n,1); g]; b = b(1:n);

%Construct graph Laplacian matrices
L = spdiags(dx, 0, n, n) - W(1:n, 1:n); % Laplacian matrix

theta = 1.2 * (2*alpha + dx*delta);   % Bound for contraction mapping.
if p == Inf
    beta = ones(size(theta));
    gamma = 1./theta;
else
    beta = (theta*p - 2)./(theta*p);
    gamma = (p-2)./(theta*p - 2);
end

% Compute Incomplete Cholesky Factorization Once:
ichol_R = ichol(L, struct('type','ict','droptol',1e-1));

% Compute initial guess:
pcg_tol = 1e-2;
u = [pcg(L, b, pcg_tol, 500, ichol_R, ichol_R'); g];

si_iter = 1e4;
err = 1e6 * ones(si_iter, 1);
for iter = 1 : si_iter

    %Compute 2-Laplacian and Infinity Laplacian
    grad_u = wnn .* (u(knn) - u(1:n));

    L2u = sum(grad_u, 2);
    LIu = min(grad_u, [], 2) + max(grad_u, [], 2);
    
    %Compute residual
    Res = alpha * L2u ./dx + delta * LIu;
    err(iter) = max(abs(Res));
    
    if err(iter) < tol; break; end
    
    %Solve linear system & attach boundary conditions
    rhs = beta .* (2*gamma.*dx.*LIu - L2u) + b;
    
    pcg_tol = min(err(iter)/100, pcg_tol);
    
    [tmp, flag, pcg_tol] = pcg(L, rhs, pcg_tol, 500, ichol_R, ichol_R', u(1:n));
    u = [tmp; g]; 
    
    % surf_plot(X, T, u, [0, 1], p)
    fprintf('Res = %.3e\n', err(iter))
end
   
end


