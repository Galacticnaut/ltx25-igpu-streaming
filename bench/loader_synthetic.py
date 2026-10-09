"""Byte-exact test of the streaming loader on a synthetic checkpoint (no model download needed).

Writes two safetensors shards of "blocks" with mixed dtypes, odd sizes and unaligned offsets (plus tensors split across
shards), then loads every block through BlockStreamer (prefetch thread) and Streamed._load_unit_direct / _read_to_gpu
(sync path), and compares against the source tensors bit for bit, including the pre-transposed Linear layout.

  .venv/bin/python bench/loader_synthetic.py [dir]
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stream"))
import torch
from safetensors.torch import save_file

from blockstream import BlockStreamer, Plan, ShardIndex
from dev import DEV, NAME
from runtime import Streamed

root = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ltx_loader_")
os.makedirs(root, exist_ok=True)
g = torch.Generator().manual_seed(0)
N_BLOCKS = 6
src = {}
for i in range(N_BLOCKS):
    src[f"blocks.{i}.attn.weight"] = torch.randn(1000 + i, 777, generator=g).to(torch.bfloat16)   # odd shape -> unaligned offsets
    src[f"blocks.{i}.attn.bias"] = torch.randn(1000 + i, generator=g).to(torch.bfloat16)
    src[f"blocks.{i}.ff.weight"] = torch.randn(3001, 513, generator=g)                             # F32 island
    src[f"blocks.{i}.scale"] = torch.randn(7, generator=g).to(torch.float16)
src["head.weight"] = torch.randn(4096, 33, generator=g).to(torch.bfloat16)                          # resident tensor
names = sorted(src)
half = len(names) // 2
save_file({k: src[k] for k in names[:half]}, os.path.join(root, "model-00001-of-00002.safetensors"))
save_file({k: src[k] for k in names[half:]}, os.path.join(root, "model-00002-of-00002.safetensors"))
print(f"device {NAME}; wrote {sum(t.numel() * t.element_size() for t in src.values()) / 2**20:.0f} MiB in 2 shards to {root}")

idx = ShardIndex(root)
assert set(idx.tensors) == set(src), "index mismatch"
bad = []


def check(name, got, transposed=False):
    want = src[name].t() if transposed else src[name]
    got = got.cpu()
    if got.dtype != want.dtype or got.shape != want.shape or not torch.equal(got.view(torch.uint8), want.contiguous().view(torch.uint8)):
        bad.append(name)


plans = []
for i in range(N_BLOCKS):
    p = Plan([t for k, t in idx.tensors.items() if k.startswith(f"blocks.{i}.")])
    if i % 2:
        p.transpose.add(f"blocks.{i}.attn.weight")   # exercise the transposed-Linear repack
    plans.append(p)

# 1. prefetch path, two passes (the stage-1 streamer cycles the model once per step)
t = time.perf_counter()
bs = BlockStreamer(plans, slots=2, passes=2)
for _ in range(2):
    for p in plans:
        s, ts = bs.next()
        for k, v in ts.items():
            check(k, v, k in p.transpose)
        bs.release(s)
bs.close()
print(f"[prefetch] {2 * N_BLOCKS} blocks in {time.perf_counter() - t:.2f} s, bytes read {bs.bytes_read / 2**20:.0f} MiB")

# 2. sync path + single-tensor reads, through Streamed's methods (no model needed for these two)
st = Streamed.__new__(Streamed)
st.idx, st.handles = idx, {}
st.abuf = torch.empty(max(p.abytes for p in plans), dtype=torch.uint8, device=DEV)
for p in plans:
    for k, v in st._load_unit_direct(p).items():
        check(k, v, k in p.transpose)
check("head.weight", st._read_to_gpu(idx.tensors["head.weight"]))
for h in st.handles.values():
    from blockstream import _close_raw
    _close_raw(h)
print("[sync] done")

print("PASS: all tensors byte-identical" if not bad else f"FAIL: {len(bad)} mismatches, e.g. {bad[:5]}")
sys.exit(1 if bad else 0)
