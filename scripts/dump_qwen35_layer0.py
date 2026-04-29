#!/usr/bin/env python3
"""Dump a single (input, output) pair for Qwen3.5 layer 0 (linear_attn).

Layer 0 of Qwen3.5 is a Gated DeltaNet block. We pick a fixed input
hidden state x (post-input_layernorm), run it through HF's
Qwen3_5GatedDeltaNet with a freshly-zeroed cache, capture the layer's
attention-path output (the `out_proj(...)` result, BEFORE residual add),
and serialise both as fp32 to a binary file.

Output format (little-endian):
    i32 magic   = 0x515E3503
    i32 hidden_size
    i32 head_v_dim
    f32 x[hidden_size]              # post-input_layernorm input
    f32 y[hidden_size]              # gated-delta output (no residual)

The Zig parity test reads this file via `valkyr --qwen35-layer-test
<model_dir> <dump_path>`, runs its CPU implementation on the same input
with zero initial state, and reports max |Δ|. The bar is ≤ 1e-3.

Usage:
    python3 scripts/dump_qwen35_layer0.py <model_dir> <out_path> [seed]

`seed` defaults to 42; pin it to make the test deterministic.
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
    seed = int(sys.argv[3]) if len(sys.argv) >= 4 else 42

    print(f"loading {model_dir} (fp32, eager)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    # `AutoModelForCausalLM` materialises the text-only trunk directly
    # at `model.model` (a Qwen3_5TextModel); no `.language_model`
    # wrapping. Layer 0 is a Qwen3_5DecoderLayer whose `.linear_attn`
    # is the Gated DeltaNet we want to test.
    text_model = model.model
    layer0 = text_model.layers[0]
    if layer0.layer_type != "linear_attention":
        raise SystemExit(f"layer 0 is {layer0.layer_type}, not linear_attention")
    linear = layer0.linear_attn
    # Loaded via AutoModelForCausalLM the config is already the text
    # submodel's config (Qwen3_5TextConfig); no extra `.text_config`
    # nesting. The multimodal route would have it.
    cfg = model.config

    hidden = cfg.hidden_size
    head_v = cfg.linear_value_head_dim
    print(f"  hidden_size = {hidden}, head_v_dim = {head_v}, num_v_heads = {cfg.linear_num_value_heads}")

    # Synthetic input. We deliberately avoid running the full embedding
    # path here so the test isolates the linear-attn math; any fixed
    # hidden state works as long as the dumper and Zig side agree.
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(hidden, dtype=np.float32) * 0.1
    x = torch.from_numpy(x_np).reshape(1, 1, hidden)

    # The HF forward switches between the chunked-parallel scan and the
    # one-step recurrent kernel based on `use_precomputed_states`. With
    # cache_params=None we get the chunked path; with a primed cache and
    # seq_len=1 we get the recurrent path. From a zero initial state
    # both produce the same output (the chunked path with chunk_size=64
    # collapses to a single step on length-1 input). We use the chunked
    # path here because priming the cache properly requires writing into
    # `last_linear_layer`'s slot and toggling `has_previous_state`,
    # which is read-only — extra plumbing for no parity gain.
    with torch.no_grad():
        y = linear(x, cache_params=None, cache_position=None)
    y_np = y.reshape(hidden).cpu().numpy().astype(np.float32, copy=False)
    print(f"  ‖x‖ = {np.linalg.norm(x_np):.4f}, ‖y‖ = {np.linalg.norm(y_np):.4f}")
    print(f"  y[:4] = {y_np[:4]}")

    out = Path(out_path)
    with out.open("wb") as f:
        f.write(struct.pack("<iii", MAGIC, hidden, head_v))
        f.write(x_np.tobytes())
        f.write(y_np.tobytes())
    print(f"  wrote {out_path} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
