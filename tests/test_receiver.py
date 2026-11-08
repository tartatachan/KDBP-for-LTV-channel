"""Check receiver interfaces, parameter count, fixed projection, and gate routing."""
import unittest
import torch
from kd_bigabp import KDBiGaBP, ReceiverConfig, receive
from kd_bigabp.gating import route_low_snr_gate, estimate_n0_from_preamble


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.config = ReceiverConfig()
        self.model = KDBiGaBP(self.config).eval()
        self.y = torch.randn(1, 70, 32, 16, dtype=torch.complex64)
        self.xp = torch.zeros(1, 8, 70, 32, dtype=torch.complex64)
        pattern = torch.randn(1, 8, 32, dtype=torch.complex64)
        for k in range(70):
            if self.config.check_is_pilot(k):
                self.xp[:, :, k] = pattern

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 306)
        self.assertEqual(set(self.model.state_dict()), {'eta_x', 'eta_h', 'beta', 'gamma'})

    def test_gate_boundary(self):
        self.assertEqual(route_low_snr_gate(3.99), 'Pilot')
        self.assertEqual(route_low_snr_gate(4.0), 'KD')
        self.assertEqual(route_low_snr_gate(float('nan')), 'Pilot')

    def test_projection(self):
        x = torch.randn(5, 32, dtype=torch.complex64)
        z = torch.fft.ifft(x, dim=-1, norm='ortho')[..., :4]
        self.assertTrue(bool((z.abs().square().sum(-1) <= x.abs().square().sum(-1) + 1e-6).all()))

    def test_both_branches(self):
        for n0, branch in [(1.0, 'Pilot'), (0.01, 'KD')]:
            soft, info = receive(self.y, self.xp, self.model, self.config, n0=n0)
            self.assertEqual(tuple(soft.shape), (1, 62, 32, 8))
            self.assertEqual(info['branch'], [branch])
            self.assertTrue(bool(torch.isfinite(soft).all()))

    def test_causal_noise_estimation(self):
        a = estimate_n0_from_preamble(self.y, self.xp, 5, self.config.alpha_true)
        changed = self.y.clone()
        changed[:, 5:] *= 100
        b = estimate_n0_from_preamble(changed, self.xp, 5, self.config.alpha_true)
        self.assertEqual(a, b)

    def test_input_validation(self):
        for n0 in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                receive(self.y, self.xp, self.model, self.config, n0=n0)
        with self.assertRaises(ValueError):
            ReceiverConfig(N=32)
        with self.assertRaises(ValueError):
            receive(self.y, self.xp, self.model, ReceiverConfig(alpha_true=0.99), n0=0.1)

    def test_frame_independence(self):
        ys = torch.cat([self.y, self.y * 0.1])
        xps = self.xp.repeat(2, 1, 1, 1)
        joint, info = receive(ys, xps, self.model, self.config)
        for i in range(2):
            separate, meta = receive(ys[i:i+1], xps[i:i+1], self.model, self.config)
            self.assertTrue(torch.equal(joint[i:i+1], separate))
            self.assertEqual(info['noise_variance'][i], meta['noise_variance'][0])
            self.assertEqual(info['branch'][i], meta['branch'][0])


if __name__ == '__main__':
    unittest.main()
