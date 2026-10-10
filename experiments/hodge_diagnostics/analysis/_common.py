"""Shared setup for the analysis scripts: run them from experiments/hodge_diagnostics as
`python analysis/<script>.py`; data are read from ./data (see hodge.fetch)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "analysis"
sys.path.insert(0, str(ROOT))
OUT.mkdir(parents=True, exist_ok=True)


def block_diag(*blocks):
    import numpy as np
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    k = 0
    for b in blocks:
        out[k:k + b.shape[0], k:k + b.shape[0]] = b
        k += b.shape[0]
    return out


def psd_sqrt(M):
    import numpy as np
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T
