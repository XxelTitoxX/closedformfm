import torch

def uniform_sampling(x0, tmin, tmax):
    t = torch.rand(x0.shape[0]).type_as(x0)[:, None]
    t = (tmax - tmin) * t + tmin
    return t # (B, 1)

# Sampling with emphasis on central time tc
def biased_sampling(x0, tmin, tmax, tc, bump_weight=0.5, bump_width=0.1):
    B = x0.shape[0]

    # choose component (B,1) so it matches t shapes
    u = torch.rand(B, device=x0.device)[:, None]   # (B,1)
    bump = u < bump_weight                         # (B,1) bool

    # uniform part (B,1)
    t_uniform = uniform_sampling(x0, tmin, tmax)   # (B,1)

    # bump part (Gaussian then clamp) also (B,1)
    sigma = (tmax - tmin) * bump_width
    tc_t = torch.as_tensor(tc, device=x0.device, dtype=x0.dtype)
    t_bump = tc_t + sigma * torch.randn(B, 1, device=x0.device, dtype=x0.dtype)
    t_bump = t_bump.clamp(tmin, tmax)

    # combine (B,1)
    t = torch.where(bump, t_bump, t_uniform)
    return t  # (B,1)