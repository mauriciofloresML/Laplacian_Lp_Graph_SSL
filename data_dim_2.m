function [X, Y, g, m] = data_dim_2(n)

% Written by Mauricio Flores, last edited on 08/22/18.

% We have decided to use three types of datasets, as follows
% (1) Two dimensional, uniform, with two labels.
% (2) High-dimensional, uniform, with few labels.
% (3) Two clusters, with different functions inside.

% This corresponds to dataset (1) within that list.
% We generate a uniform distribution in [0,1] x [0,1], 
% with labeled points at the extremes.

% Unlabeled points
X = rand(n, 2); 

% Labeled points
m = 2;
g = [0; 1]; 
Y = [0.5*ones(2, 1), [0; 1]];

end