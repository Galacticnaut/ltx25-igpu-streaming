"""Mac port, step 1: can the LTX-2 transformer run on Apple MPS at all? No checkpoint needed (random weights).

Builds the diffusers LTX2VideoTransformer3DModel with the default (LTX-2) dims but only N_LAYERS blocks, then:
  1. float64 RoPE (rope_double_precision=True, the diffusers default) on MPS -> does it error?
  2. float32 RoPE on MPS vs float64 RoPE on CPU -> how different are the RoPE tables?
  3. bf16 forward on MPS: finite? relative diff vs a CPU fp32 reference with identical weights
  4. time per block at stage-1 (~3k video tokens) and stage-2 (~12k) sizes -> per-step estimate for 48 blocks

  .venv/bin/python bench/mps_probe.py            (PYTORCH_ENABLE_MPS_FALLBACK=1 to let missing ops run on CPU)
"""
import os
import time
import traceback

import torch
from diffusers import LTX2VideoTransformer3DModel

N_LAYERS = int(os.environ.get("N_LAYERS", 2))
TEXT = 512           # MAXSEQ used by run_ltx.py
FPS = 12.0
dev = torch.device("mps")


def build(double):
    torch.manual_seed(0)
    m = LTX2VideoTransformer3DModel(num_layers=N_LAYERS, rope_double_precision=double)
    return m.eval().requires_grad_(False)


def inputs(F, h, w, a_frames, dtype, device):
    """Packed latents as the pipeline feeds them: video [1, F*h*w, 128], audio [1, a_frames, 128]."""
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g).to(device, dtype)
    return dict(hidden_states=r(1, F * h * w, 128), audio_hidden_states=r(1, a_frames, 128),
                encoder_hidden_states=r(1, TEXT, 3840), audio_encoder_hidden_states=r(1, TEXT, 3840),
                timestep=torch.full((1,), 900.0, device=device), audio_timestep=torch.full((1,), 900.0, device=device),
                sigma=torch.full((1,), 0.9, device=device), audio_sigma=torch.full((1,), 0.9, device=device),
                encoder_attention_mask=torch.ones(1, TEXT, device=device), audio_encoder_attention_mask=torch.ones(1, TEXT, device=device),
                num_frames=F, height=h, width=w, fps=FPS, audio_num_frames=a_frames, return_dict=False)


def run(m, kw):
    out = m(**kw)
    return out[0], out[1]


def stats(name, v, a, ref=None):
    s = f"{name}: video finite={bool(torch.isfinite(v).all())} std={v.float().std():.4f}  audio finite={bool(torch.isfinite(a).all())} std={a.float().std():.4f}"
    if ref is not None:
        rv = ((v.float().cpu() - ref[0]).norm() / ref[0].norm()).item()
        ra = ((a.float().cpu() - ref[1]).norm() / ref[1].norm()).item()
        s += f"  rel diff vs CPU fp32: video {rv:.4f} audio {ra:.4f}"
    print(s, flush=True)


def timed(m, kw, reps=3):
    run(m, kw); torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        v, a = run(m, kw)
    torch.mps.synchronize()
    return (time.perf_counter() - t) / reps, v, a


print(f"torch {torch.__version__}, MPS available={torch.backends.mps.is_available()}, fallback={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')}")
blk = sum(p.numel() for p in build(False).transformer_blocks[0].parameters())
print(f"block params {blk / 1e6:.0f} M = {blk * 2 / 2**20:.0f} MiB bf16 (the 22B checkpoint's blocks are 738 MiB)\n")

# FLASH stage 1: 768x448, 65 frames -> latent 9 x 14 x 24 = 3024 tokens; audio 5.4 s at 25 latent frames/s ~ 136
S1 = (9, 14, 24, 136)
S2 = (9, 28, 48, 136)       # x2 latent upsample -> 12096 tokens

# 1. float64 RoPE on MPS
print("[1] float64 RoPE (diffusers default) on MPS, bf16")
try:
    m = build(True).to(dev, torch.bfloat16)
    v, a = run(m, inputs(*S1, torch.bfloat16, dev)); torch.mps.synchronize()
    stats("    ran OK", v, a)
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
del m

# 2. RoPE tables: float32 on MPS vs float64 on CPU
print("\n[2] RoPE tables, float32 on MPS vs float64 on CPU")
try:
    m64, m32 = build(True), build(False).to(dev)
    F, h, w, af = S2
    vc = m64.rope.prepare_video_coords(1, F, h, w, torch.device("cpu"), fps=FPS)
    c64 = m64.rope(vc, device=torch.device("cpu"))
    c32 = m32.rope(vc.to(dev), device=dev)
    for i, (x, y) in enumerate(zip(c64, c32)):
        print(f"    table {i}: max abs diff {(x.double() - y.cpu().double()).abs().max():.2e}")
    del m64, m32
except Exception:
    print("    could not compare directly:"); traceback.print_exc(limit=2)

# 3. bf16 MPS vs fp32 CPU, same weights, stage-1 size
print("\n[3] numerics at stage-1 size (bf16 MPS vs fp32 CPU, float64 RoPE on CPU)")
cpu = build(True).float()
t = time.perf_counter()
ref = [x.float() for x in run(cpu, inputs(*S1, torch.float32, torch.device("cpu")))]
print(f"    CPU fp32 reference: {time.perf_counter() - t:.1f} s")
m = build(False)
m.load_state_dict(cpu.state_dict())
m = m.to(dev, torch.bfloat16)
del cpu
v, a = run(m, inputs(*S1, torch.bfloat16, dev)); torch.mps.synchronize()
stats("    MPS bf16", v, a, ref)

# 4. timing
print(f"\n[4] timing (bf16, {N_LAYERS} blocks; extrapolated to 48 blocks = 1 transformer pass = 1 denoising step)")
for name, S in (("stage 1 (3024 video tok)", S1), ("stage 2 (12096 video tok)", S2)):
    try:
        dt, v, a = timed(m, inputs(*S, torch.bfloat16, dev))
        per = dt / N_LAYERS
        print(f"    {name}: {per * 1000:.0f} ms/block -> ~{per * 48:.1f} s/step compute (iGPU measured ~47 s / ~240 s)"
              f"  finite={bool(torch.isfinite(v).all() and torch.isfinite(a).all())}  peak MPS mem {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB")
    except Exception as e:
        print(f"    {name}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}")
