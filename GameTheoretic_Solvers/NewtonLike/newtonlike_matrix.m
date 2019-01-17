% The objective of this code is to compute the next iteration. 
% Depending on the size of the matrix, one might consider using GMRES, 
% combined with an ILU preconditioner, in order to speedup convergence.
% NOTE: The linear system is not symmetric.

function u = newtonlike_matrix(grad_u, g, knn, wnn, dx, p, n, m, ...
                    k_neigh, iterative, u_prev, err_prev)

if nargin < 10; iterative = 0; end  % We just solve with backslash.                            
                            
% 'u_prev'  : is the previous iteration, useful to converge faster.
% 'err_prev': is the previous residual, useful to pick an appropriate
%             tolerance for GMRES. I recommend to divide by 100 or 1000.

% Compute indexes where max/min occur:
[~, index_max] = max(grad_u, [], 2);
[~, index_min] = min(grad_u, [], 2);

% Maybe can be vectorized, but it is fast enough.
beta = ones(n, k_neigh); p_2 = p-2;
for i = 1 : n
    beta(i, index_max(i)) = 1 + p_2 * dx(i);
    beta(i, index_min(i)) = 1 + p_2 * dx(i);
end

% Assemble A & B:
A = sparse((1:n)'*ones(1, k_neigh), knn, -beta.*wnn);
A = A - [diag(sum(A,2)), zeros(n,m)];
B = A(:, n+1:end);

if iterative == 0
    u = [-A(:, 1:n) \ (B*g); g];
    
elseif iterative == 1
    
    % We use an Incomplete LU factorization as the preconditioner.
    % Both the results and the speed are sensitive to the choice of
    % droptol, and I would encourage the users to choose their own.
    % In my experience, larger droptol is faster but less accurate.
    % A droptol = 1e-1 might result in a zero pivot, breaking ILU.
    
    setup.udiag = 1;
    setup.type = 'crout';
    setup.droptol = 1e-1;
    
    [L, U] = ilu(A(:, 1:n), setup);
    
    u = [gmres(-A(:, 1:n), B*g, 20, err_prev/100, 200, L, U, u_prev); g];
        
end

end