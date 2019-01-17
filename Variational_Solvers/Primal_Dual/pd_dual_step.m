function v = pd_dual_step(knn, c, v, ubar, idx, r1, q, choice, bisec_iter) 

if nargin == 7; choice = 1; end         % By default, use gradient descent.
if nargin == 8; bisec_iter = 50; end    % By default, 50 bisection steps.

%NOTE: c = w_conj(idx). This is a constant, so it has been computed before.

% Define parameters a, b: (idx picks points where c is nonzero)
a = ubar(knn)-ubar; a = a(idx); b = v(idx); 

if choice == 1  % This is one step of gradient descent. Fast but inexact.
    v(idx) = v(idx) - r1 * (a + c .* abs(b).^(q-1) .* sign(b));
    % Replacing r1 with a smaller step could work too. 
    
elseif choice == 2 % This is bisection solver, slow but precise.
    v(idx) = pd_dual_bisection(a, b, c, r1, q, bisec_iter);
end

end
    