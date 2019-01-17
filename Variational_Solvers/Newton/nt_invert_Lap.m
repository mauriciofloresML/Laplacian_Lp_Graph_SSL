function [Lap_Bg] = nt_invert_Lap(wnn, knn, u, g, n, m, k_neigh, p, iterative, err)

% By default, use an iterative solver
if nargin == 8; iterative = 1; end

% Eliminate weights between labeled points (if needed)
wnn = wnn(1:n,:); knn = knn(1:n,:);

%% Compute 'Lap':
A_Values = wnn .* abs(u(1:n) - u(knn)).^(p-2);
A = sparse((1:n)'*ones(1, k_neigh), knn, A_Values, n, n+m);

B = A(:, n+1:end); A = A(:,1:n);

% Fully vectorized computation of Matrix 'D'
D = spdiags(sum(A,2) + sum(B,2), 0, n, n);

% Compute matrix 'Lap' (which we will invert)
Lap = D - A; 

%% Invert:
% For d = 2, use iterative = 0.
% For d > 2, use iterative = 1 (much faster, especially once d > 5)

if iterative == 0
    Lap_Bg = Lap \ (B*g);
    
else
    
    droptol = 1e-1; % Choosing 1e-2 might be more accurate, but slower. 
    
    % Compute Incomplete Cholesky
    ichol_R = ichol(Lap, struct('type','ict','droptol',droptol));
    
    % Iterative CG solver with with IC preconditioner
    Lap_Bg = pcg(Lap, B*g, err/100, 500, ichol_R, ichol_R', u(1:n));
end

end