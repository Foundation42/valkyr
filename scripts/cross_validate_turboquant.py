#!/usr/bin/env python3
"""
Cross-validate the Zig TQ4 quantizer against YATQ Python reference.

Given a deterministic 256-element input vector, run YATQ's
quantize_value(bits=4) and emit the resulting indices as a Zig
[256]u8 literal plus a few scalars (raw L2 norm, reconstruction
norm). The Zig side hardcodes the same input + expected indices
in a smoke test and verifies bit-exact parity.

Run from the repo root:
    python3 scripts/cross_validate_turboquant.py

Requires reference/turboquant/YATQ/ to exist (gitignored upstream
clone of github.com/arclabs001/YATQ).
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference", "turboquant", "YATQ"))
from turboquant_wht import TurboQuantWHT  # noqa: E402

D = 256
x = torch.tensor([(i / 128.0) - 1.0 for i in range(D)], dtype=torch.float32)

tq = TurboQuantWHT(dim=D, bits=4)
res = tq.quantize_value(x.unsqueeze(0))
indices = res["indices"][0].numpy().astype(np.int64)
vec_norm = float(res["vec_norm"][0].item())

recon = tq.reconstruct_key(res)[0].numpy().astype(np.float32)
recon_norm = float(np.linalg.norm(recon))
gamma_correction = vec_norm / recon_norm

print(f"// YATQ reference values (b=4) for x[i] = (i/128.0) - 1.0, i in [0,256).")
print(f"// raw_norm        = {vec_norm:.8f}")
print(f"// recon_norm(raw) = {recon_norm:.8f}")
print(f"// gamma_corrected = {gamma_correction:.8f}")
print()
print(f"const yatq_indices_b4 = [256]u8{{")
for row_start in range(0, D, 16):
    chunk = indices[row_start : row_start + 16]
    print("    " + ", ".join(f"{int(v):2d}" for v in chunk) + ",")
print(f"}};")
print()
print(f"// First 8 of reconstruction with RAW norm scaling (YATQ style):")
print(
    "// "
    + ", ".join(f"{float(v):+.6f}" for v in recon[:8])
)
print()
print(f"// L2 norm of YATQ reconstruction (raw-norm rescaled): {recon_norm:.8f}")
print(f"// (Norm-correction γ would give exactly raw_norm = {vec_norm:.8f}.)")
