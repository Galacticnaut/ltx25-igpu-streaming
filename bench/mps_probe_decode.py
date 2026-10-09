"""Mac port, step 1b: do the non-transformer components run on Apple MPS? Random weights, default diffusers configs.

Mirrors how the engine uses them (decode_big.py, run_ltx.py):
  video VAE      bf16, tiled decode of the FLASH stage-2 latent (1536x896, 65 frames); encode of a 768x448 start image (I2V)
  latent upsampler bf16, x2 on the stage-1 latent (768x448 -> 1536x896)
  audio VAE      fp32 decode of a 5.4 s audio latent
  vocoder (BWE)  fp32, mel -> 48 kHz stereo
Each: CPU fp32 vs MPS relative diff at a small size, then full size on MPS (time, finite, peak memory).

  .venv/bin/python bench/mps_probe_decode.py          (PYTORCH_ENABLE_MPS_FALLBACK=1 to let missing ops run on CPU)
"""
import os
import time

import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

dev, cpu = torch.device("mps"), torch.device("cpu")


def rnd(*s):
    return torch.randn(*s, generator=torch.Generator().manual_seed(1))


def sync():
    torch.mps.synchronize()


def first_line(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:220]}"


def compare(name, build, fn, x_small, dtype):
    """Same weights on CPU fp32 and on MPS in `dtype`; print relative diff of fn(model, x)."""
    try:
        torch.manual_seed(0)
        m = build().eval().requires_grad_(False)
        with torch.no_grad():
            ref = fn(m, x_small).float()
            m = m.to(dev, dtype)
            out = fn(m, x_small.to(dev, dtype)).float().cpu()
        rel = ((out - ref).norm() / ref.norm()).item()
        print(f"  {name} numerics: out {tuple(out.shape)} finite={bool(torch.isfinite(out).all())} rel diff vs CPU fp32 {rel:.4f}", flush=True)
    except Exception as e:
        print(f"  {name} numerics: FAILED {first_line(e)}", flush=True)


def full(name, build, fn, x, dtype):
    try:
        torch.manual_seed(0)
        m = build().eval().requires_grad_(False).to(dev, dtype)
        torch.mps.empty_cache()
        with torch.no_grad():
            t = time.perf_counter()
            out = fn(m, x.to(dev, dtype)); sync()
            dt = time.perf_counter() - t
        print(f"  {name} full size: in {tuple(x.shape)} -> out {tuple(out.shape)} {dt:.1f} s finite={bool(torch.isfinite(out).all())}"
              f" peak MPS mem {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB", flush=True)
        del m, out
    except Exception as e:
        print(f"  {name} full size: FAILED {first_line(e)}", flush=True)
    torch.mps.empty_cache()


def vae():
    # the default config leaves the ratios unset and tiling then scales tiles wrongly (also on CPU); the checkpoint config sets them
    m = AutoencoderKLLTX2Video(spatial_compression_ratio=32, temporal_compression_ratio=8)
    m.enable_tiling()
    return m


def vae_decode(m, z):
    ts = torch.tensor([0.0], device=z.device, dtype=z.dtype) if m.config.timestep_conditioning else None
    return m.decode(z, ts, return_dict=False)[0]


def vae_encode(m, x):
    return m.encode(x).latent_dist.mode()


def audio_decode(m, z):
    return m.decode(z, return_dict=False)[0]


print(f"torch {torch.__version__}, fallback={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')}\n")

print("[video VAE, bf16]")
compare("decode", vae, vae_decode, rnd(1, 128, 2, 4, 6), torch.bfloat16)
full("decode", vae, vae_decode, rnd(1, 128, 9, 28, 48), torch.bfloat16)          # FLASH stage 2 -> 1536x896, 65 frames
compare("encode", vae, vae_encode, rnd(1, 3, 1, 128, 192).clamp(-1, 1), torch.bfloat16)
full("encode", vae, vae_encode, rnd(1, 3, 1, 448, 768).clamp(-1, 1), torch.bfloat16)  # I2V start frame at stage-1 size

print("\n[latent upsampler, bf16]")
compare("x2", LTX2LatentUpsamplerModel, lambda m, z: m(z), rnd(1, 128, 2, 4, 6), torch.bfloat16)
full("x2", LTX2LatentUpsamplerModel, lambda m, z: m(z), rnd(1, 128, 9, 14, 24), torch.bfloat16)

print("\n[audio VAE, fp32]")
compare("decode", AutoencoderKLLTX2Audio, audio_decode, rnd(1, 8, 16, 16), torch.float32)
full("decode", AutoencoderKLLTX2Audio, audio_decode, rnd(1, 8, 136, 16), torch.float32)   # ~5.4 s

print("\n[vocoder with BWE, fp32]")
mel = lambda T: rnd(1, 2, T, 64) * 0.5
compare("mel->wav", LTX2VocoderWithBWE, lambda m, x: m(x), mel(64), torch.float32)
full("mel->wav", LTX2VocoderWithBWE, lambda m, x: m(x), mel(544), torch.float32)          # 5.4 s of 100 Hz mel frames
