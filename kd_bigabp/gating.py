"""Estimate preamble noise and select KD-BiGaBP or pilot-only processing per frame."""
from __future__ import annotations
import math
import torch
from .pilot import pilot_only

def snr_db_from_n0(n0_hat: float, es: float=1.0) -> float:
    return 10.0 * math.log10(float(es) / max(float(n0_hat), 1e-12))

def route_low_snr_gate(snr_est_db: float, threshold_db: float=4.0) -> str:
    if not math.isfinite(snr_est_db):
        return 'Pilot'
    equal = math.isclose(float(snr_est_db), float(threshold_db), rel_tol=0.0, abs_tol=1e-10)
    return 'KD' if float(snr_est_db) >= float(threshold_db) or equal else 'Pilot'

def estimate_n0_from_preamble(
    Y_frame: torch.Tensor, Xp_frame: torch.Tensor, Kp1: int,
    alpha_assumed: float | None = None, n0_floor: float = 0.0001,
) -> tuple[float, float]:
    if Y_frame.ndim == 4:
        assert Y_frame.shape[0] == 1, 'Noise estimation must be strictly single-frame (no batch pooling)'
        Yp = Y_frame[0, :Kp1]
    else:
        Yp = Y_frame[:Kp1]
    if Xp_frame.ndim == 4:
        Xp0 = Xp_frame[0, :, 0, :]
    else:
        Xp0 = Xp_frame[:, 0, :]
    if Kp1 < 2:
        raise ValueError('Repeated-preamble noise estimator requires Kp1 >= 2')
    diffs = Yp[1:] - Yp[:-1]
    (n_sub, n_rx) = (Yp.shape[1], Yp.shape[2])
    n0_blind = float(diffs.abs().square().sum().item() / (2.0 * (Kp1 - 1) * n_sub * n_rx))
    n0_blind = max(n0_blind, n0_floor)
    if alpha_assumed is None:
        return (n0_blind, n0_blind)
    pilot_energy_per_sub_rx = float(Xp0.abs().square().sum().item() / (n_sub * n_rx))
    drift_bias = (1.0 - float(alpha_assumed)) * pilot_energy_per_sub_rx
    n0_comp = max(n0_blind - drift_bias, n0_floor)
    return (n0_blind, n0_comp)

@torch.no_grad()
def receive(Y, Xp, model, config, n0=None, threshold_db=4.0):
    model_config = (model.M, model.N, model.N_sub, model.L_taps, model.Es, model.alpha_ar1)
    requested = (config.M, config.N, config.N_sub, config.L_taps, config.Es, config.alpha_true)
    if model_config != requested:
        raise ValueError('The model and receiver configuration must use identical dimensions and priors.')
    if Y.device.type != 'cpu' or Xp.device.type != 'cpu' or next(model.parameters()).device.type != 'cpu':
        raise ValueError('This reference receiver is validated on CPU.')
    if Y.ndim != 4 or tuple(Y.shape[1:]) != (config.K_time, config.N_sub, config.N):
        raise ValueError('Y must have shape [batch, 70, 32, 16].')
    if tuple(Xp.shape) != (Y.shape[0], config.M, config.K_time, config.N_sub) or Y.shape[0] < 1:
        raise ValueError('Xp must have shape [batch, 8, 70, 32].')
    if Y.dtype != torch.complex64 or Xp.dtype != torch.complex64:
        raise ValueError('Y and Xp must use complex64.')
    if not torch.isfinite(Y).all() or not torch.isfinite(Xp).all():
        raise ValueError('Receiver inputs must be finite.')
    if not math.isfinite(threshold_db):
        raise ValueError('threshold_db must be finite.')
    if n0 is not None and (not math.isfinite(float(n0)) or float(n0) <= 0):
        raise ValueError('n0 must be finite and positive.')
    pred, estimates, branches, channels = [], [], [], []
    for i in range(Y.shape[0]):
        y, xp = Y[i:i+1], Xp[i:i+1]
        if n0 is None:
            if not torch.equal(xp[:, :, :config.Kp1], xp[:, :, :1].expand(-1, -1, config.Kp1, -1)):
                raise ValueError('Blind noise estimation requires identical repeated preamble pilots.')
            _, noise = estimate_n0_from_preamble(y, xp, config.Kp1, config.alpha_true)
        else:
            noise = float(n0)
        branch = route_low_snr_gate(snr_db_from_n0(noise, config.Es), threshold_db)
        if branch == 'KD':
            soft, info = model(y, xp, noise, config, return_info=True)
        else:
            soft, info = pilot_only(y, xp, noise, config)
        pred.append(soft)
        channels.append(info['channel_estimate'])
        estimates.append(noise)
        branches.append(branch)
    return torch.cat(pred), {
        'channel_estimate': torch.cat(channels),
        'noise_variance': estimates,
        'branch': branches,
    }
