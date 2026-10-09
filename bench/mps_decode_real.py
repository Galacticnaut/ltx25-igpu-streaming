"""Mac port: real-weight decode components on MPS. Audio VAE + vocoder in fp32 vs bf16 (the bf16 vocoder gave NaN on XPU, B19),
CPU fp32 reference; video VAE (bf16, tiled) decode of a FLASH-size latent: finite, time, memory.

  .venv/bin/python bench/mps_decode_real.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stream"))
import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from dev import DEV, MODELS as M, acc

g = torch.Generator().manual_seed(3)
z_audio = torch.randn(1, 8, 136, 16, generator=g)          # normalized audio latent, ~5.4 s
cpu = torch.device("cpu")


def audio(dev, vae_dt, voc_dt):
    av = AutoencoderKLLTX2Audio.from_pretrained(f"{M}/audio_vae", torch_dtype=vae_dt).to(dev).eval()
    vo = LTX2VocoderWithBWE.from_pretrained(f"{M}/vocoder", torch_dtype=voc_dt).to(dev).eval()
    with torch.no_grad():
        t = time.perf_counter()
        mel = av.decode(z_audio.to(dev, vae_dt), return_dict=False)[0]
        wav = vo(mel.to(voc_dt))
        if dev.type == "mps": acc.synchronize()
        return mel.float().cpu(), wav.float().cpu(), time.perf_counter() - t


def stat(tag, mel, wav, ref=None, dt=None):
    s = f"  {tag:28s} mel finite={bool(torch.isfinite(mel).all())}  wav finite={bool(torch.isfinite(wav).all())} " \
        f"NaN frac {torch.isnan(wav).float().mean():.3f}  rms {wav.nan_to_num().pow(2).mean().sqrt():.4f}"
    if ref is not None and torch.isfinite(wav).all():
        s += f"  rel diff vs CPU fp32 {((wav - ref).norm() / ref.norm()).item():.4f}"
    if dt is not None: s += f"  {dt:.1f} s"
    print(s, flush=True)


print("[audio VAE + vocoder, real weights]")
mel_c, wav_c, dt = audio(cpu, torch.float32, torch.float32); stat("CPU fp32/fp32 (reference)", mel_c, wav_c, dt=dt)
mel, wav, dt = audio(DEV, torch.float32, torch.float32); stat("MPS fp32/fp32 (engine path)", mel, wav, wav_c, dt)
mel, wav, dt = audio(DEV, torch.float32, torch.bfloat16); stat("MPS fp32 VAE / bf16 vocoder", mel, wav, wav_c, dt)
mel, wav, dt = audio(cpu, torch.float32, torch.bfloat16); stat("CPU fp32 VAE / bf16 vocoder", mel, wav, wav_c, dt)
acc.empty_cache()

print("\n[video VAE, real weights, bf16, tiled, FLASH stage-2 latent -> 65 x 1536x896]")
vae = AutoencoderKLLTX2Video.from_pretrained(f"{M}/vae", torch_dtype=torch.bfloat16).to(DEV).eval(); vae.enable_tiling()
z = torch.randn(1, 128, 9, 28, 48, generator=g).to(DEV, torch.bfloat16)
with torch.no_grad():
    ts = torch.tensor([0.0], device=DEV, dtype=torch.bfloat16) if vae.config.timestep_conditioning else None
    t = time.perf_counter(); v = vae.decode(z, ts, return_dict=False)[0]; acc.synchronize()
print(f"  out {tuple(v.shape)} finite={bool(torch.isfinite(v).all())} range [{v.min().item():.2f}, {v.max().item():.2f}] "
      f"{time.perf_counter() - t:.1f} s, MPS driver mem {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB "
      f"(allocated {torch.mps.current_allocated_memory() / 2**30:.1f} GiB)")
