"""Mac port: error build-up through all 48 REAL transformer blocks, MPS bf16 (engine path) vs CPU fp32.

A fp32 copy of the 22B model does not fit in RAM, so the reference is replayed block by block on the CPU:
  1. run the real transformer on MPS exactly like run_ltx.py (streamed blocks, pre-transposed Linear weights, CPU RoPE,
     zero-filled missing params), recording each block's inputs and outputs;
  2. replay the 48 blocks on the CPU in fp32 as an independent chain from the same block-0 inputs;
  3. per block: "chain" = MPS vs CPU-fp32 chain (accumulated error), "local" = one CPU fp32 block on MPS's own input
     (that block's own bf16 error).
Inputs are random latents / text states at FLASH stage-1 size (3024 video tokens), sigma 0.9.

  .venv/bin/python bench/mps_drift_real.py
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stream"))
import torch
from diffusers import LTX2VideoTransformer3DModel
from safetensors import safe_open

from dev import DEV, MODELS, acc
from runtime import Streamed, build_empty

F, h, w, AF, TEXT = 9, 14, 24, 136, 512
TDIR = f"{MODELS}/transformer"
T0 = time.perf_counter()


def log(*a):
    print(f"[{time.perf_counter() - T0:6.1f}s]", *a, flush=True)


def tree(x, fn):
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, (tuple, list)):
        return type(x)(tree(v, fn) for v in x)
    if isinstance(x, dict):
        return {k: tree(v, fn) for k, v in x.items()}
    return x


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---------- 1. MPS run with the engine's streaming path ----------
dit = build_empty(lambda: LTX2VideoTransformer3DModel.from_config(LTX2VideoTransformer3DModel.load_config(TDIR)))
units = [f"transformer_blocks.{i}" for i in range(48)]
ds = Streamed(dit, TDIR, units, prefetch_passes=1, slots=2, log=log, transpose_linear=True)
ds.load_resident()
for name, p in dit.named_parameters():
    if p.device.type == "meta" and not name.startswith("transformer_blocks."):
        path, _, leaf = name.rpartition(".")
        (dit.get_submodule(path) if path else dit)._parameters[leaf] = torch.nn.Parameter(torch.zeros(p.shape, dtype=p.dtype, device=DEV), requires_grad=False)

cap = {"in0": None, "out": []}


def pre(mod, args, kwargs):
    if mod._unit_i == 0:
        cap["in0"] = (tree(args, lambda t: t.detach().float().cpu()), tree(kwargs, lambda t: t.detach().float().cpu()))


def post(mod, args, kwargs, out):
    cap["out"].append(tuple(o.detach().float().cpu() for o in out))


for u in units:
    m = dit.get_submodule(u)
    m.register_forward_pre_hook(pre, with_kwargs=True)
    m.register_forward_hook(post, with_kwargs=True)

g = torch.Generator().manual_seed(1)
r = lambda *s: torch.randn(*s, generator=g).to(DEV, torch.bfloat16)
kw = dict(hidden_states=r(1, F * h * w, 128), audio_hidden_states=r(1, AF, 128),
          encoder_hidden_states=r(1, TEXT, 4096), audio_encoder_hidden_states=r(1, TEXT, 2048),
          timestep=torch.full((1,), 900.0, device=DEV), audio_timestep=torch.full((1,), 900.0, device=DEV),
          sigma=torch.full((1,), 0.9, device=DEV), audio_sigma=torch.full((1,), 0.9, device=DEV),
          encoder_attention_mask=torch.ones(1, TEXT, device=DEV), audio_encoder_attention_mask=torch.ones(1, TEXT, device=DEV),
          num_frames=F, height=h, width=w, fps=12.0, audio_num_frames=AF, return_dict=False)
log("MPS forward, 48 streamed blocks...")
t = time.perf_counter()
with torch.no_grad():
    vout, aout = dit(**kw)
acc.synchronize()
log(f"MPS forward {time.perf_counter() - t:.1f} s (streamed {ds.bytes / 2**30:.1f} GiB, waited {ds.wait_s:.1f} s on I/O); "
    f"final finite video={bool(torch.isfinite(vout).all())} audio={bool(torch.isfinite(aout).all())}")
assert len(cap["out"]) == 48, len(cap["out"])
ds.close()
del dit, ds
acc.empty_cache()

# ---------- 2./3. CPU fp32 replay ----------
template = build_empty(lambda: LTX2VideoTransformer3DModel.from_config(
    {**LTX2VideoTransformer3DModel.load_config(TDIR), "num_layers": 1})).transformer_blocks[0]
template = template.to("meta")
from blockstream import ShardIndex
idx = ShardIndex(TDIR)
files = {}


def cpu_block(i):
    b = copy.deepcopy(template)
    sd = {}
    for k, t in idx.tensors.items():
        if k.startswith(f"transformer_blocks.{i}."):
            f = files.get(t.file) or files.setdefault(t.file, safe_open(t.file, "pt"))
            sd[k[len(f"transformer_blocks.{i}."):]] = f.get_tensor(k).float()
    b.load_state_dict(sd, assign=True, strict=True)
    return b.eval().requires_grad_(False)


args0, kwargs0 = cap["in0"]


def call(b, v, a):
    if "hidden_states" in kwargs0:
        return b(*args0, **{**kwargs0, "hidden_states": v, "audio_hidden_states": a})
    return b(v, a, *args0[2:], **kwargs0)


v_in = kwargs0.get("hidden_states", args0[0] if args0 else None)
a_in = kwargs0.get("audio_hidden_states", args0[1] if len(args0) > 1 else None)
chain = (v_in, a_in)
mps_prev = (v_in, a_in)
print(f"\n{'block':>5} | {'chain video':>11} {'chain audio':>11} | {'local video':>11} {'local audio':>11} | finite")
worst_local = 0.0
with torch.no_grad():
    for i in range(48):
        b = cpu_block(i)
        chain = call(b, *chain)
        local = call(b, *mps_prev)
        mo = cap["out"][i]
        cv, ca = rel(mo[0], chain[0]), rel(mo[1], chain[1])
        lv, la = rel(mo[0], local[0]), rel(mo[1], local[1])
        worst_local = max(worst_local, lv, la)
        fin = all(bool(torch.isfinite(x).all()) for x in mo)
        if i < 4 or i % 4 == 3 or not fin:
            print(f"{i:5d} | {cv:11.4f} {ca:11.4f} | {lv:11.4f} {la:11.4f} | {fin}", flush=True)
        mps_prev = mo
        del b
print(f"\nfinal chain rel diff: video {cv:.4f} audio {ca:.4f}; worst single-block (local) diff {worst_local:.4f}")
log("done")
