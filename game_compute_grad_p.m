function grad_p = game_compute_grad_p(grad_u, dx, p)

if p == Inf
    grad_p = max(grad_u, [], 2) + min(grad_u, [], 2);
else
    grad_2 = sum(grad_u, 2) ./ dx;
    grad_inf = max(grad_u, [], 2) + min(grad_u, [], 2);
    grad_p = grad_2 / p + grad_inf * (1 - 2/p);
end
    
end