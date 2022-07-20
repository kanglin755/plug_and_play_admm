import torch

def pnp_admm(
        measurements, forward, forward_adjoint, denoiser, 
        step_size=1e-4, num_iter=50, max_cgiter=100, cg_tol=1e-7
    ):
    """
    ADMM plug and play
    """
    x_h =  forward_adjoint(measurements)

    def conjugate_gradient(A, b, x0, max_iter, tol):
        """
        Conjugate gradient method for solving Ax=b
        """
        x = x0
        r = b-A(x)
        d = r
        for _ in range(max_iter):
            z = A(d)
            rr = torch.sum(r**2)
            alpha = rr/torch.sum(d*z)
            x += alpha*d
            r -= alpha*z
            if torch.norm(r)/torch.norm(b) < tol:
                break
            beta = torch.sum(r**2)/rr
            d = r + beta*d        
        return x

    def cg_leftside(x):
        """
        Return left side of Ax=b, i.e., Ax
        """
        return forward_adjoint(forward(x)) + step_size*x

    def cg_rightside(x):
        """
        Returns right side of Ax=b, i.e. b
        """
        return x_h + step_size*x

    # Start
    x = torch.zeros_like(x_h)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    for _ in range(num_iter):
        b = cg_rightside(v-u)
        x = conjugate_gradient(cg_leftside, b, x, max_cgiter, cg_tol)
        v = denoiser(x+u)
        u += (x - v)
    return v
