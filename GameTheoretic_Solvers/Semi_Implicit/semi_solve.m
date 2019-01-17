function [u, res, iter] = semi_solve(W, g, p, knn, wnn, dx, n, tol)

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

% Factorize Laplacian matrix:
[R, flag, S] = chol(L);
assert(flag == 0, 'Bad Cholesky Factorization')

% Compute initial guess:
u = [S * (R\(R'\(S'*b))); g];

for iter = 1 : 15000
    
    %Compute 2-Laplacian and Infinity Laplacian
    grad_u = wnn .* (u(knn) - u(1:n));

    L2u = sum(grad_u, 2);
    LIu = min(grad_u, [], 2) + max(grad_u, [], 2);
    
    %Compute residual
    grad_p = alpha * L2u ./dx + delta * LIu;
    res = max(abs(grad_p));
    if res < tol; break; end
    
    %Solve linear system & attach boundary conditions
    rhs = beta .* (2*gamma.*dx.*LIu - L2u) + b;
    
    % u = [L \ rhs; g];                % Direct inversion of 'L'
    u = [S * (R\(R'\(S'*rhs))); g];    % Cholesky inversion (much faster)
    
    fprintf('Iter = %d, Res = %.3e\n', iter, res)
end
   
end


