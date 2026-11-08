"""Implement the 306-parameter KD-BiGaBP receiver and fixed delay-domain feedback."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .config import ReceiverConfig

class KDBiGaBP(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        g = config or ReceiverConfig()
        (self.M, self.N, self.N_sub, self.L_taps) = (g.M, g.N, g.N_sub, g.L_taps)
        self.T_iter = 10
        self.Es = g.Es
        self.Phi = float(g.tap_powers.sum() / g.N_sub)
        self.cx = np.sqrt(g.Es / 2)
        self.sigmoid_gain = 4.0
        self.alpha_ar1 = float(g.alpha_true)
        self.eta_x = nn.Parameter(torch.zeros(g.N, 9))
        self.eta_h = nn.Parameter(torch.zeros(g.N, 9))
        self.beta = nn.Parameter(torch.ones(1, 9))
        self.gamma = nn.Parameter(torch.ones(1, 9))
        self.eps = 1e-06
        self.register_buffer('tap_power', g.tap_powers, persistent=False)
        self.register_buffer('process_variance', (1 - self.alpha_ar1 ** 2) * g.tap_powers, persistent=False)

    def run_bigabp(self, t, x_hat, psi_x, h_hat, psi_h, y_k, N0_est, h_prior_mu, h_prior_phi):
        zeta_x_t = torch.sigmoid(self.sigmoid_gain * self.eta_x[:, t]).view(1, 1, 1, self.N)
        zeta_h_t = torch.sigmoid(self.sigmoid_gain * self.eta_h[:, t]).view(1, self.N, 1, 1)
        gamma_t = torch.relu(self.gamma[0, t])
        beta_t = torch.relu(self.beta[0, t])
        x_hat_p = x_hat.permute(0, 3, 2, 1)
        pure_prod = h_hat * x_hat_p
        sum_prod = torch.sum(pure_prod, dim=3, keepdim=True)
        y_tilde = y_k.view(-1, self.N, 1, 1) - (sum_prod - pure_prod)
        psi_x_p = psi_x.permute(0, 3, 2, 1)
        psi_h_p = psi_h
        term = torch.abs(h_hat) ** 2 * psi_x_p + (torch.abs(x_hat_p) ** 2 + psi_x_p) * psi_h_p
        nu_y = torch.clamp(torch.sum(term, dim=3, keepdim=True) - term + N0_est, min=self.eps)
        nu_x = nu_y + self.Es * psi_h_p
        tmp_denom = torch.clamp(nu_x, min=self.eps)
        tmp_num = torch.abs(h_hat) ** 2
        psi_r = 1.0 / torch.clamp(
            torch.sum(tmp_num / tmp_denom, dim=1, keepdim=True) - tmp_num / tmp_denom,
            min=self.eps,
        )
        tmp_val = h_hat.conj() * y_tilde / tmp_denom
        r_hat = psi_r * (torch.sum(tmp_val, dim=1, keepdim=True) - tmp_val)
        r_hat_for_x = r_hat.permute(0, 3, 2, 1)
        psi_r_for_x = psi_r.permute(0, 3, 2, 1)
        norm_factor = self.cx * (psi_r_for_x + self.eps)
        x_var_real = torch.tanh(gamma_t * r_hat_for_x.real / norm_factor)
        x_var_imag = torch.tanh(gamma_t * r_hat_for_x.imag / norm_factor)
        x_var = self.cx * torch.complex(x_var_real, x_var_imag)
        x_hat_new = zeta_x_t * x_var + (1 - zeta_x_t) * x_hat
        psi_x_new = torch.clamp(zeta_x_t * (self.Es - torch.abs(x_var) ** 2) + (1 - zeta_x_t) * psi_x, min=0.0)
        nu_h = torch.clamp(nu_y + self.Phi * psi_x_p, min=self.eps)
        weight_t = beta_t
        psi_q = 1.0 / torch.clamp(
            torch.sum(weight_t * torch.abs(x_hat_p) ** 2 / nu_h, dim=2, keepdim=True) + 1e-10,
            min=self.eps,
        )
        term_q_all = weight_t * x_hat_p.conj() * y_tilde / nu_h
        q_hat = psi_q * torch.sum(term_q_all, dim=2, keepdim=True)
        mu_prior = h_prior_mu
        phi_prior = torch.clamp(h_prior_phi, min=self.eps)
        psi_post = 1.0 / (1.0 / phi_prior + 1.0 / psi_q)
        mu_post = psi_post * (mu_prior / phi_prior + q_hat / psi_q)
        h_hat_new = zeta_h_t * mu_post + (1 - zeta_h_t) * h_hat
        psi_h_new = torch.clamp(zeta_h_t * psi_post + (1 - zeta_h_t) * psi_h, min=0.0)
        return (x_hat_new, psi_x_new, h_hat_new, psi_h_new)

    def forward_step(self, y, xpilot, is_pilot, mu, var, n0):
        B = y.shape[0]
        if mu is None:
            mu = torch.zeros(B, self.N, self.M, self.L_taps, dtype=y.dtype, device=y.device)
            var = self.tap_power.view(1, 1, 1, -1).expand_as(mu.real).clone()
        else:
            mu = self.alpha_ar1 * mu
            var = self.alpha_ar1 ** 2 * var + self.process_variance.view(1, 1, 1, -1)
        hprior = torch.fft.fft(mu, n=self.N_sub, dim=-1, norm='ortho')
        fvar = (var.sum(-1, keepdim=True) / self.N_sub).expand(-1, -1, -1, self.N_sub)
        hp = hprior.permute(0, 3, 1, 2).reshape(-1, self.N, 1, self.M)
        vp = fvar.permute(0, 3, 1, 2).reshape(-1, self.N, 1, self.M)
        yy = y.reshape(-1, self.N).unsqueeze(-1)
        xx = xpilot.reshape(-1, self.M).view(-1, self.M, 1, 1).expand(-1, -1, -1, self.N)
        xh = xx.clone()
        vx = torch.zeros_like(xh.real) if is_pilot else torch.ones_like(xh.real) * self.Es
        hh = hp.clone()
        vh = vp.clone()
        for t in range(self.T_iter - 1):
            (xh, vx, hh, vh) = self.run_bigabp(t, xh, vx, hh, vh, yy, n0, hp, vp)
            if is_pilot:
                xh = xx.clone()
                vx = torch.zeros_like(xh.real)
        hf = hh.squeeze(2).view(B, self.N_sub, self.N, self.M).permute(0, 2, 3, 1)
        post = torch.fft.ifft(hf, dim=-1, norm='ortho')[..., :self.L_taps]
        pv = 0.9 * var
        pred = xh.mean(3).squeeze(2).view(B, self.N_sub, self.M)
        return (pred, post, pv)

    def forward(self, Y, Xpilots, n0, g, return_info=False):
        mu = var = None
        outputs = []
        hs = []
        for k in range(g.K_time):
            isp = g.check_is_pilot(k)
            xp = (
                Xpilots[:, :, k].transpose(-1, -2) if isp
                else torch.zeros(Y.shape[0], g.N_sub, g.M, dtype=Y.dtype, device=Y.device)
            )
            (pred, mu, var) = self.forward_step(Y[:, k], xp, isp, mu, var, n0)
            if not isp:
                outputs.append(pred)
                if return_info:
                    hs.append(torch.fft.fft(mu, n=g.N_sub, dim=-1, norm='ortho').permute(0, 3, 1, 2))
        pred = torch.stack(outputs, 1)
        return (pred, {'channel_estimate': torch.stack(hs, 1)}) if return_info else pred
