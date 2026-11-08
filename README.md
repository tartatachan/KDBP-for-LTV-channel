# KD-BiGaBP for Wideband LTV Channel Tracking

PyTorch code for **Resolving the Dimensionality–Coherence Constraint:
Data-Aided Tracking for Wideband LTV Channels**.

Includes the KD-BiGaBP receiver, delay-domain feedback, and low-SNR gating
with a pilot-only fallback.

## Usage

Install the dependencies and run the CPU example and tests:

```bash
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```
