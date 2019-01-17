function [idx, w_conj2] = pd_idx_to_solve(knn, wnn, q)

% The objective of this code is to find the indexes that for 
% which the dual-step solve is necessary (we want to save time) 

% Two criteria need to be satisfied (explained in notes)
% 
% 1.- The weights w_{xy} cannot be zero.
% 2.- The diagonal (x = y) cannot be included.

idx1 = find(wnn);
idx2 = find( (1:size(knn,1))' - knn);

idx = intersect(idx1, idx2);
w_conj2 = wnn(idx).^(1-q);

% We could save even more computations by using the skew-symmetric
% property of the dual variable, but this creates new issues, due
% to unstructured memory access.

end