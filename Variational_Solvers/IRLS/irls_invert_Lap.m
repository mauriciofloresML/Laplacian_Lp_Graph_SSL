function [Lap_Bg] = irls_invert_Lap(w, knn, u, g, n, m, k_neigh, p)

% Eliminate weights between labeled points (if needed)
w = w(1:n,:); knn = knn(1:n,:);

%% Compute 'Lap':
A_Values = w .* abs(u(1:n) - u(knn)).^(p-2);
A = sparse((1:n)'*ones(1, k_neigh), knn, A_Values, n, n+m);

B = A(:, n+1:end); A = A(:,1:n);

% Fully vectorized computation of Matrix 'D'
D = spdiags(sum(A,2) + sum(B,2), 0, n, n);

Lap = D - A; 

%% Invert:
Lap_Bg = Lap \ (B*g);

% Since this method isn't very good, we did not bother
% to include an iterative solver (but if you wish to do
% so, just look up "newton_solve.m" inside Newton folder.

end
