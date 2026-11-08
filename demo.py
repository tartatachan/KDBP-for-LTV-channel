"""Run a nominal receiver interface check on synthetic input tensors."""
import argparse
import torch
from kd_bigabp import KDBiGaBP, ReceiverConfig, receive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = ReceiverConfig()
    model = KDBiGaBP(config).eval()
    if args.weights:
        state = torch.load(args.weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state.get('state_dict', state), strict=True)
    pilots = torch.zeros(1, 8, 70, 32, dtype=torch.complex64)
    pattern = torch.randn(1, 8, 32, dtype=torch.complex64)
    for k in range(config.K_time):
        if config.check_is_pilot(k):
            pilots[:, :, k] = pattern
    received = torch.randn(1, 70, 32, 16, dtype=torch.complex64)
    for noise in (1.0, 10**(-12/10), None):
        soft, info = receive(received, pilots, model, config, n0=noise)
        print('branch:', info['branch'], 'soft shape:', tuple(soft.shape),
              'finite:', bool(torch.isfinite(soft).all()))
    print('Interface check only; these inputs are not a BER experiment.')
    if not args.weights:
        print('Using initialization values, not trained manuscript weights.')


if __name__ == '__main__':
    main()
