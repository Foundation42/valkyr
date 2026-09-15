#!/usr/bin/env python3
"""Concatenate .vtr traces into one, for concept-drift / phase-change probes.

The seam between two workloads is a hard phase change: instruction working set,
data working set and access pattern all switch at once.
"""
import struct, sys

def main(out, parts):
    total = 0
    with open(out, "wb") as f:
        f.write(b"VTRACE01" + struct.pack("<QQ", 0, 0))
        for p in parts:
            with open(p, "rb") as g:
                head = g.read(24)
                assert head[:8] == b"VTRACE01", p
                n = struct.unpack("<Q", head[8:16])[0]
                left = n * 8
                while left:
                    chunk = g.read(min(1 << 22, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
                total += n
        f.seek(8)
        f.write(struct.pack("<Q", total))
    print(f"{out}: {total} refs from {len(parts)} traces")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
