#!/usr/bin/env python3
"""HF transformers ↔ Zig CPU bit-close parity test for Qwen3.

Runs HuggingFace transformers Qwen3-4B-Instruct-2507 forward in fp32 on
a single BOS token and dumps argmax + top-5 + the max logit. The Zig
side (`valkyr --gen <Qwen3 dir> 151643`) is then compared by hand —
argmax must match, top-5 IDs must overlap, max logit |Δ| must be small
(target ≤3.0, in line with the Gemma 2B parity bar).

Usage:
    python3 scripts/cross_validate_qwen3.py [<model dir>]

If no dir is given, picks the canonical HF cache snapshot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    if len(sys.argv) >= 2:
        model_dir = sys.argv[1]
    else:
        cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
        snapshots = list(cache.iterdir())
        if not snapshots:
            print("no Qwen3-4B snapshot in HF cache; pass model dir as argv[1]", file=sys.stderr)
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

    # Qwen3 has no canonical BOS; tok.bos_token_id is None. We test with
    # `<|endoftext|>` (151643), which is the EOS / padding marker — it's
    # the same token the Zig --gen smoke uses, so the parity bar is
    # apples-to-apples.
    bos_id = 151643
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
    print("expected Zig CPU output (from `valkyr --gen <model> 151643`):")
    print("  top 5 logits:")
    print("    id=   220  logit=    8.2430  Ġ")
    print("    id=151643  logit=    8.0166  <|endoftext|>")
    print("    id=   334  logit=    7.9441  **")
    print("    id=   198  logit=    7.8595  Ċ")
    print("    id=    13  logit=    7.6558  .")
    print()
    print("parity bar: argmax matches, top-5 IDs are a permutation, max |Δ| ≤ 3.0.")


if __name__ == "__main__":
    main()
