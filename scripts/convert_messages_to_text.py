#!/usr/bin/env python3
"""Convert chat-style {"messages": [...]} JSONL to {"text": "..."} JSONL
with the Qwen3 chat template applied.

Usage:
    python convert_messages_to_text.py --in pack_native.jsonl --out pack_native_text.jsonl

The Qwen3 chat template (matches `tokenizer.apply_chat_template` for
Qwen/Qwen3-0.6B) is:

    <|im_start|>{role}
    {content}<|im_end|>

with a trailing newline between turns. We do not append the
"<|im_start|>assistant\n" generation prompt — this is training data,
the assistant turn is already a full message.
"""

import argparse
import json
from pathlib import Path


def render(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    args = ap.parse_args()

    n_in = 0
    n_out = 0
    char_lens: list[int] = []

    with args.src.open("r") as src, args.dst.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            messages = row["messages"]
            text = render(messages)
            char_lens.append(len(text))
            dst.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_out += 1

    if char_lens:
        char_lens.sort()
        mean = sum(char_lens) / len(char_lens)
        med = char_lens[len(char_lens) // 2]
        p95 = char_lens[int(0.95 * len(char_lens))]
        print(
            f"converted {n_in} → {n_out} rows; "
            f"text chars min/p50/mean/p95/max = "
            f"{char_lens[0]}/{med}/{mean:.0f}/{p95}/{char_lens[-1]}"
        )
    else:
        print(f"converted {n_in} → {n_out} rows")


if __name__ == "__main__":
    main()
