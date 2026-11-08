"""Define the nominal QPSK receiver configuration and statistical channel prior."""
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class ReceiverConfig:
    M: int = 8
    N: int = 16
    N_sub: int = 32
    L_taps: int = 4
    Kp1: int = 5
    Kd1: int = 31
    Kp2: int = 3
    Kd2: int = 31
    Es: float = 1.0
    alpha_true: float = 0.9960560528944907

    def __post_init__(self):
        dimensions = (self.M, self.N, self.N_sub, self.L_taps)
        frame = (self.Kp1, self.Kd1, self.Kp2, self.Kd2)
        if dimensions + frame + (self.Es,) != (8, 16, 32, 4, 5, 31, 3, 31, 1.0):
            raise ValueError('This release supports the nominal 8x16, 32-subcarrier QPSK configuration only.')
        if not math.isfinite(self.alpha_true) or abs(self.alpha_true) > 1:
            raise ValueError('alpha_true must be finite and within [-1, 1].')

    @property
    def K_time(self):
        return self.Kp1 + self.Kd1 + self.Kp2 + self.Kd2

    @property
    def tap_powers(self):
        p = np.exp(-0.5 * np.arange(self.L_taps, dtype=np.float64))
        return torch.tensor(p / p.sum() * self.N_sub / self.N, dtype=torch.float32)

    def check_is_pilot(self, k):
        return k < self.Kp1 or self.Kp1 + self.Kd1 <= k < self.Kp1 + self.Kd1 + self.Kp2
