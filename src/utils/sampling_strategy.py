import torch

def uniform_sampling(x0, tmin, tmax):
    t = torch.rand(x0.shape[0]).type_as(x0)[:, None]
    t = (tmax - tmin) * t + tmin
    return t # (B, 1)

# Sampling with emphasis on central time tc
def biased_sampling(x0, tmin, tmax, tc, bump_weight=0.5, bump_width=0.1):
    B = x0.shape[0]

    # choose component
    u = torch.rand(B, device=x0.device)
    bump = u < bump_weight

    # uniform part
    t_uniform = uniform_sampling(x0, tmin, tmax)  # (B,)

    # bump part (Gaussian then clamp)
    sigma = (tmax - tmin) * bump_width
    tc_t = torch.as_tensor(tc, device=x0.device, dtype=x0.dtype)
    t_bump = tc_t + sigma * torch.randn(B, device=x0.device, dtype=x0.dtype)
    t_bump = torch.clamp(t_bump, tmin, tmax)

    # combine
    t = torch.where(bump, t_bump, t_uniform)  # all (B,)
    return t[:, None]  # (B, 1)