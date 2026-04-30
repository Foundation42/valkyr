#!/usr/bin/env python3
"""HF transformers ↔ Zig CPU bit-close parity test for Qwen3.5.

Runs HuggingFace transformers Qwen3.5 forward in fp32 on a single
endoftext token and dumps argmax + top-5 + the max logit. The Zig
side (`valkyr --gen <Qwen3.5 dir> 248044`) is then compared by hand —
argmax must match, top-5 IDs must overlap, max logit |Δ| must be small
(≤ 3.0, in line with the Gemma 2B / Qwen3-4B parity bar).

Usage:
    python3 scripts/cross_validate_qwen35.py [<model_dir>]

Defaults to the local Qwen3.5-0.8B snapshot — the smallest variant we
have, fastest to run end-to-end on a CPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force the torch fallback paths for both the optional CUDA causal_conv1d
# extension (broken local install) and the FLA kernels. The torch
# implementations are bit-identical to the recurrent decode that valkyr
# implements, so this preserves parity.
import transformers.utils.import_utils as _iu
_iu.is_causal_conv1d_available = lambda: False
_iu.is_flash_linear_attention_available = lambda: False

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    if len(sys.argv) >= 2:
        model_dir = sys.argv[1]
    else:
        cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots"
        snapshots = list(cache.iterdir())
        if not snapshots:
            print("no Qwen3.5-0.8B snapshot in HF cache; pass model dir as argv[1]", file=sys.stderr)
            sys.exit(1)
        model_dir = str(snapshots[0])

    print(f"loading model from: {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    # Qwen3.5 has no canonical BOS; the canonical "context-free" probe
    # is the EOS / endoftext marker (id 248044). Same one Zig uses.
    bos_id = 248044
    print(f"input token id: {bos_id}  ({tok.decode([bos_id])!r})")

    input_ids = torch.tensor([[bos_id]], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0, 0].cpu().numpy()  # [vocab_size]

    print()
    print(f"vocab_size: {logits.shape[0]}")
    print(f"max logit:  {float(logits.max()):.4f}")
    print(f"argmax id:  {int(logits.argmax())}  ({tok.decode([int(logits.argmax())])!r})")
    print()
    print("HF transformers top-5 logits:")
    top_idx = (-logits).argsort()[:5]
    for i in top_idx:
        s = tok.decode([int(i)])
        print(f"  id={int(i):>6}  logit={float(logits[i]):>10.4f}  {s!r}")

    print()
    print("expected Zig CPU output (from `valkyr --gen <model> 248044`):")
    print("  argmax = 266 ('at'), max logit ≈ 16.45")
    print()
    print("parity bar: argmax matches, top-5 IDs are a permutation, max |Δ| ≤ 3.0.")


if __name__ == "__main__":
    main()
