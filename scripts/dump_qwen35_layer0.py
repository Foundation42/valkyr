#!/usr/bin/env python3
"""Dump a single (input, output) pair for one Qwen3.5 layer.

Picks a fixed synthetic post-input_layernorm hidden state `x`, runs it
through the chosen layer's ATTENTION-PATH (linear_attn or self_attn,
NOT including the residual add or the FFN), and writes both to a binary
file the Zig parity test can read.

Output format (little-endian):
    i32 magic           = 0x515E3503
    i32 hidden_size
    i32 layer_idx
    i32 layer_type_kind  # 0 = linear_attention, 1 = full_attention
    f32 x[hidden_size]   # post-input_layernorm input
    f32 y[hidden_size]   # layer attention-path output (no residual)

The Zig parity test reads this file via:
    valkyr --qwen35-layer-test <model_dir> <dump_path>
runs its CPU implementation on the same input (with empty cache) and
reports max |Δ|. The bar is ≤ 1e-3.

Usage:
    python3 scripts/dump_qwen35_layer0.py <model_dir> <out_path> [layer_idx] [seed]

Defaults: `layer_idx=0` (the first linear-attn layer), `seed=42`. To
exercise the full-attention path, pass `layer_idx=3` (the first full
layer in the standard 3:1 schedule).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Force the torch fallback paths for causal_conv1d_update and the fused
# linear-attention kernel. Both have native CUDA extensions that fail to
# import on this machine (ABI skew vs the local CUDA runtime); the torch
# implementations are bit-identical for the recurrent (decode) step,
# which is what we test against.
import transformers.utils.import_utils as _iu
_iu.is_causal_conv1d_available = lambda: False
_iu.is_flash_linear_attention_available = lambda: False

from transformers import AutoModelForCausalLM


MAGIC = 0x515E3503


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    model_dir = sys.argv[1]
    out_path = sys.argv[2]
    layer_idx = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    seed = int(sys.argv[4]) if len(sys.argv) >= 5 else 42

    print(f"loading {model_dir} (fp32, eager)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    text_model = model.model
    layer = text_model.layers[layer_idx]
    cfg = model.config
    hidden = cfg.hidden_size
    print(f"  hidden_size = {hidden}, layer_idx = {layer_idx}, layer_type = {layer.layer_type}")

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(hidden, dtype=np.float32) * 0.1
    x = torch.from_numpy(x_np).reshape(1, 1, hidden)

    if layer.layer_type == "linear_attention":
        # Linear-attn (Gated DeltaNet). With cache_params=None the
        # chunked-parallel path runs; from a zero initial state, on a
        # length-1 input, it's mathematically identical to the
        # recurrent decode (which is what Zig implements).
        kind = 0
        with torch.no_grad():
            y = layer.linear_attn(x, cache_params=None, cache_position=None)
    elif layer.layer_type == "full_attention":
        # Full attention. Build cos/sin from the rotary embedding at
        # position 0 (single token, empty KV cache → single-element
        # softmax = 1.0). The `Qwen3_5TextRotaryEmbedding` is
        # constructed inside the text_model and cached; we just call it.
        kind = 1
        position_ids = torch.arange(0, 1, dtype=torch.long).unsqueeze(0)
        rotary = text_model.rotary_emb
        cos, sin = rotary(x, position_ids)
        with torch.no_grad():
            y, _ = layer.self_attn(
                hidden_states=x,
                position_embeddings=(cos, sin),
                attention_mask=None,
                past_key_values=None,
                cache_position=position_ids[0],
            )
    else:
        raise SystemExit(f"unsupported layer_type: {layer.layer_type}")

    y_np = y.reshape(hidden).cpu().numpy().astype(np.float32, copy=False)
    print(f"  ‖x‖ = {np.linalg.norm(x_np):.4f}, ‖y‖ = {np.linalg.norm(y_np):.4f}")
    print(f"  y[:4] = {y_np[:4]}")

    out = Path(out_path)
    with out.open("wb") as f:
        f.write(struct.pack("<iiii", MAGIC, hidden, layer_idx, kind))
        f.write(x_np.tobytes())
        f.write(y_np.tobytes())
    print(f"  wrote {out_path} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
