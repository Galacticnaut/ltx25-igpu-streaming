"""Page-cache A/B for the streaming loader: read every transformer block PASSES times through BlockStreamer (no compute) and report
GB/s per pass. Run once with LTX_NOCACHE=1 (default, F_NOCACHE: bypass the macOS cache) and once with LTX_NOCACHE=0.

  LTX_NOCACHE=0 .venv/bin/python bench/io_cache_ab.py [passes]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stream"))
import psutil
from blockstream import BlockStreamer, Plan, ShardIndex
from dev import MODELS

PASSES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
idx = ShardIndex(f"{MODELS}/transformer")
plans = [Plan([t for k, t in idx.tensors.items() if k.startswith(f"transformer_blocks.{i}.")]) for i in range(48)]
gib = sum(p.nbytes for p in plans) / 2**30
bs = BlockStreamer(plans, slots=2, passes=PASSES)
print(f"LTX_NOCACHE={os.environ.get('LTX_NOCACHE', '1')}: {gib:.1f} GiB per pass, free RAM {psutil.virtual_memory().available / 2**30:.1f} GiB")
for p in range(PASSES):
    t = time.perf_counter()
    for _ in plans:
        s, _t = bs.next(); bs.release(s)
    dt = time.perf_counter() - t
    print(f"  pass {p + 1}: {dt:5.1f} s = {gib / dt:5.2f} GiB/s   free RAM after {psutil.virtual_memory().available / 2**30:5.1f} GiB", flush=True)
bs.close()
