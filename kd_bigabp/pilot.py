"""Provide causal pilot-only channel tracking and QPSK soft detection for fallback."""
from __future__ import annotations
import numpy as np
import torch

def init_filter(g, batch):
    D = g.M * g.L_taps
    power = g.tap_powers.repeat(g.M)
    P0 = torch.diag(power).to(torch.complex64).expand(batch, -1, -1).clone()
    mu = torch.zeros(batch, D, g.N, dtype=torch.complex64, device=g.tap_powers.device)
    return (mu, P0.clone(), P0)

def channel(mu, g):
    h = mu.transpose(-1, -2).reshape(mu.shape[0], g.N, g.M, g.L_taps)
    return torch.fft.fft(h, n=g.N_sub, dim=-1, norm='ortho').permute(0, 3, 1, 2)

def lmmse_update(mu, cov, x, y, n0, dft, selected):
    (B, Ns, Nt) = x.shape
    A = (x[..., None] * dft[None, :, None, :]).reshape(B, Ns, -1)
    A = A * selected[..., None]
    obs = y * selected[..., None]
    cross = cov @ A.mH
    S = A @ cross + float(n0) * torch.eye(Ns, dtype=A.dtype, device=x.device)
    gain = torch.linalg.solve(S, cross.mH).mH
    updated = mu + gain @ (obs - A @ mu)
    J = torch.eye(cov.shape[-1], dtype=cov.dtype, device=x.device) - gain @ A
    post = J @ cov @ J.mH + float(n0) * gain @ gain.mH
    return (updated, (post + post.mH) / 2)

def _qpsk_constellation(device: torch.device, dtype: torch.dtype=torch.complex64, es: float=1.0) -> torch.Tensor:
    scale = float(np.sqrt(es / 2.0))
    real = torch.tensor([1.0, 1.0, -1.0, -1.0], device=device)
    imag = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)
    return torch.complex(real, imag).to(dtype) * scale

def _constellation_mmse_denoiser_torch(
    z: torch.Tensor, tau: float, es: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    const = _qpsk_constellation(z.device, z.dtype, es=es)
    tau_t = torch.as_tensor(float(tau), dtype=torch.float32, device=z.device).clamp_min(1e-12)
    dist = torch.abs(z.unsqueeze(-1) - const.view(*(1,) * z.ndim, 4)) ** 2
    metric = -dist / tau_t
    metric = metric - torch.max(metric, dim=-1, keepdim=True).values
    probs = torch.softmax(metric, dim=-1)
    mean = torch.sum(probs.to(z.dtype) * const.view(*(1,) * z.ndim, 4), dim=-1)
    second = torch.sum(probs * (torch.abs(const) ** 2).view(*(1,) * z.ndim, 4), dim=-1)
    var = torch.clamp(second - torch.abs(mean) ** 2, min=0.0)
    return (mean, var)

def _delay_dft_matrix_torch(generator, device: torch.device) -> torch.Tensor:
    f = torch.arange(generator.N_sub, dtype=torch.float32, device=device).view(-1, 1)
    l = torch.arange(generator.L_taps, dtype=torch.float32, device=device).view(1, -1)
    angle = -2.0 * np.pi * f * l / float(generator.N_sub)
    return torch.complex(torch.cos(angle), torch.sin(angle)) / np.sqrt(float(generator.N_sub))

def _mmse_detect_one(Y_block: torch.Tensor, H_block: torch.Tensor, N0: float, generator) -> torch.Tensor:
    (batch_size, n_sub, n_rx) = Y_block.shape
    n_tx = generator.M
    H_flat = H_block.reshape(batch_size * n_sub, n_rx, n_tx)
    y_flat = Y_block.reshape(batch_size * n_sub, n_rx, 1)
    H_h = H_flat.conj().transpose(1, 2)
    gram = torch.matmul(H_h, H_flat)
    eye = torch.eye(n_tx, dtype=H_flat.dtype, device=Y_block.device).unsqueeze(0)
    rhs = torch.matmul(H_h, y_flat)
    x_flat = torch.linalg.solve(gram + float(N0) * eye, rhs).squeeze(-1)
    return x_flat.view(batch_size, n_sub, n_tx)

def pilot_only(Y, Xpilots, n0, g):
    if Y.device.type != 'cpu':
        raise ValueError('The validated fallback in this reference release supports CPU only.')
    B, K, Ns, Nr = Y.shape
    mu, cov, P0 = init_filter(g, B)
    dft = _delay_dft_matrix_torch(g, Y.device)
    pred, hs = [], []
    for k in range(K):
        if k:
            mu = float(g.alpha_true) * mu
            cov = float(g.alpha_true)**2 * cov + (1 - float(g.alpha_true)**2) * P0
        if g.check_is_pilot(k):
            mu, cov = lmmse_update(
                mu, cov, Xpilots[:, :, k].transpose(-1, -2), Y[:, k], n0, dft,
                torch.ones(B, Ns, device=Y.device),
            )
            continue
        hh = channel(mu, g)
        x0 = _mmse_detect_one(Y[:, k], hh, n0, g)
        pred.append(_constellation_mmse_denoiser_torch(x0, n0)[0])
        hs.append(hh)
    return torch.stack(pred, 1), {'channel_estimate': torch.stack(hs, 1)}
