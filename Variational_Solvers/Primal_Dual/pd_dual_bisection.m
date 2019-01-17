function x = pd_dual_bisection(a, b, c, r1, q, bisec_iter)

% Define alpha & identify relevant indexes:
alpha = b - a*r1; idx_relevant = find(alpha);

% Now define 'alpha_pos' and 'beta' only on relevant domain:
alpha_pos = abs(alpha(idx_relevant)); 
beta = 2 * c(idx_relevant) * r1;

q_1 = q-1; % Save a tiny bit of time:

% Define function to solve:
func = @(x) 2*x - 2*alpha_pos + beta .* x.^(q_1); 

% Define search range for positive x:
xmin = 0; xmax = alpha_pos;

% Check range is correct!
total = sign ( func(xmin) .* func(xmax) ) + 1;
if max(total) > 0 
    disp('Warning! Bad Range');
    disp('We stop here!!')
end

for i = 1 : bisec_iter    
    x_temp = 0.5 * (xmin + xmax);
    
    % A very efficient way to update xmax/xmin:
    vec = 0.5*(1+sign(func(x_temp)));
    xmax = (1-vec).*xmax + vec.*x_temp;
    xmin = vec.*xmin + (1-vec).*x_temp;    
end

x_pos = 0.5 * (xmax + xmin);

% Now, convert Solution Into Signed Solution:
x = zeros(size(a));
x(idx_relevant) = x_pos .* sign(alpha(idx_relevant));

end
 
