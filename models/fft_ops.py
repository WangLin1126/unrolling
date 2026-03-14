"""FFT-based convolution and data-consistency operations."""

import torch


def fft_conv2d_circular(x: torch.Tensor, otf: torch.Tensor) -> torch.Tensor:
    """Circular convolution via FFT.

    Args:
        x:   (B, C, H, W) real image
        otf: (B, H, W//2+1) or (1, H, W//2+1) complex OTF

    Returns:
        (B, C, H, W) convolved image
    """
    X = torch.fft.rfft2(x)
    if otf.dim() == 3:
        otf = otf.unsqueeze(1)
    Y = X * otf
    return torch.fft.irfft2(Y, s=x.shape[-2:])


def fft_data_step(x_t: torch.Tensor, v: torch.Tensor,
                  otf: torch.Tensor, beta: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """Closed-form frequency-domain u-update (Wiener-like).

    Solves:  u = argmin_u  0.5 ||g * u - x_t||^2  +  beta/2 ||u - v||^2

    U(w) = [ conj(G(w)) X_t(w) + beta V(w) ] / [ |G(w)|^2 + beta ]
    """
    X_t = torch.fft.rfft2(x_t)
    V = torch.fft.rfft2(v)

    if otf.dim() == 3:
        otf = otf.unsqueeze(1)

    G_conj = otf.conj()
    G_abs2 = (otf * G_conj).real

    if isinstance(beta, torch.Tensor) and beta.dim() >= 1:
        while beta.dim() < 4:
            beta = beta.unsqueeze(-1)

    numerator = G_conj * X_t + beta * V
    denominator = G_abs2 + beta + eps

    U = numerator / denominator
    return torch.fft.irfft2(U, s=x_t.shape[-2:])


def gaussian_otf(delta: torch.Tensor, H: int, W: int,
                 device=None, dtype=torch.float32) -> torch.Tensor:
    """Compute the OTF of a 2D Gaussian with std=delta analytically.

    G(wx, wy) = exp(-0.5 * delta^2 * (wx^2 + wy^2))

    Args:
        delta: scalar (0-dim) or batched (B,) blur std
    Returns:
        (1, H, W//2+1) if scalar, (B, H, W//2+1) if batched
    """
    if not isinstance(delta, torch.Tensor):
        delta = torch.tensor(delta, device=device, dtype=dtype)
    device = delta.device
    dtype = delta.dtype

    wy = torch.fft.fftfreq(H, device=device).to(dtype) * 2 * torch.pi
    wx = torch.fft.rfftfreq(W, device=device).to(dtype) * 2 * torch.pi
    WY, WX = torch.meshgrid(wy, wx, indexing="ij")  # (H, W//2+1)

    if delta.dim() >= 1:
        # delta (B,) → (B, 1, 1) for broadcasting with (H, W//2+1)
        d2 = (delta.view(-1, 1, 1)) ** 2
        otf = torch.exp(-0.5 * d2 * (WX ** 2 + WY ** 2))  # (B, H, W//2+1)
    else:
        otf = torch.exp(-0.5 * delta ** 2 * (WX ** 2 + WY ** 2))
        otf = otf.unsqueeze(0)  # (1, H, W//2+1)
    return otf.to(torch.complex64)