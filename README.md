# Laplacian_Lp_Graph_SSL

Email questions to mauricio.a.flores.math@gmail.com if
there's anything unclear about the codes, happy to answer.

* Make sure to keep files in their respective folder, since
some files inside the folder call files outside the folder,
so if they are moved, the reference will no longer work. *

* Details of the paper and implementation can be found in the
paper "Algorithms for Lp-based semi-supervised learning on graphs".

Most notations in the codes follow the standards in the paper:

'n' is the number of unlabeled data points
'm' is the number of labeled data points
'g' is the labeling function (see paper)
'u' is the function we need to solve for
'k_neigh' is the number of neighbors.


'knn' is a matrix of size 'n' by 'k_neigh' (or 'n+m' by
'k_neigh'), which maps which vertices are connected to which.

'wnn' is the same size as 'knn', and indicates the nonzero-weights.
