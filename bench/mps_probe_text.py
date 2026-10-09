"""Mac port, step 1c: Gemma-4 Unified text encoder + LTX-2 text connectors on Apple MPS. Random weights, no checkpoint.

Gemma is built with the 12B text dims from BIBLE.md (3840 hidden, k=v on global layers; head counts are guesses until the real config is readable) but only N_LAYERS layers
(5 sliding-window + 1 global by default, so both attention kinds run). 512 tokens, left padding, as run_ltx.py uses.
Connectors: per-modality projections of the 49 stacked hidden states (188,160 features/token), 8 blocks each.
Each: CPU fp32 vs MPS bf16 relative diff, finite check, time; Gemma time is extrapolated to 48 layers.

  .venv/bin/python bench/mps_probe_text.py
"""
import os
import time

import torch
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from transformers.models.gemma4_unified import Gemma4UnifiedTextConfig, Gemma4UnifiedTextModel

N_LAYERS = int(os.environ.get("N_LAYERS", 6))
SEQ, PAD = 512, 400            # MAXSEQ, and a short prompt: 112 real tokens, 400 left-pad
dev, cpu = torch.device("mps"), torch.device("cpu")


def first_line(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:220]}"


def rel(a, b):
    return ((a.float().cpu() - b.float()).norm() / b.float().norm()).item()


def sync():
    torch.mps.synchronize()


# ---------- Gemma ----------
cfg = Gemma4UnifiedTextConfig(hidden_size=3840, intermediate_size=15360, num_hidden_layers=N_LAYERS,
                              num_attention_heads=16, num_key_value_heads=8, head_dim=256, global_head_dim=512,
                              attention_k_eq_v=True, num_global_key_value_heads=4, sliding_window=1024,
                              layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * (N_LAYERS // 6) + ["sliding_attention"] * (N_LAYERS % 6))
ids = torch.randint(3, 262144, (1, SEQ), generator=torch.Generator().manual_seed(1))
mask = torch.ones(1, SEQ, dtype=torch.long)
mask[:, :PAD] = 0
ids[:, :PAD] = 0


def gemma_states(m, device):
    out = m(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True, use_cache=False)
    return torch.stack(out.hidden_states, -1)          # (1, SEQ, 3840, N_LAYERS + 1); the pipeline stacks all 49


print(f"torch {torch.__version__}\n\n[Gemma-4 Unified text encoder, {N_LAYERS} layers, {SEQ} tokens ({SEQ - PAD} real)]")
try:
    torch.manual_seed(0)
    m = Gemma4UnifiedTextModel(cfg).eval().requires_grad_(False)
    print(f"  params {sum(p.numel() for p in m.parameters()) / 1e9:.2f} B (embedding table {m.embed_tokens.weight.numel() / 1e9:.2f} B)")
    with torch.no_grad():
        t = time.perf_counter()
        ref = gemma_states(m, cpu)
        print(f"  CPU fp32 reference {time.perf_counter() - t:.1f} s")
        m = m.to(dev, torch.bfloat16)
        gemma_states(m, dev); sync()
        t = time.perf_counter()
        hs = gemma_states(m, dev); sync()
        dt = time.perf_counter() - t
    real = mask[0].bool()
    per_layer = [rel(hs[0, real, :, i], ref[0, real, :, i]) for i in range(hs.shape[-1])]
    print(f"  MPS bf16: finite={bool(torch.isfinite(hs[0, real]).all())}  rel diff per hidden state (real tokens): "
          + " ".join(f"{d:.3f}" for d in per_layer))
    print(f"  MPS time {dt:.2f} s for {N_LAYERS} layers -> ~{dt / N_LAYERS * 48:.1f} s for 48 (iGPU phase A+B measured ~20 s incl. streaming)"
          f"  peak MPS mem {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB")
    del m, hs, ref
except Exception as e:
    print(f"  FAILED {first_line(e)}")
torch.mps.empty_cache()

# ---------- connectors ----------
print("\n[LTX-2 text connectors, 8 blocks per modality]")
emb = torch.randn(1, SEQ, 3840 * 49, generator=torch.Generator().manual_seed(2)) * 0.05
for double in (True, False):
    tag = "float64 RoPE (default)" if double else "float32 RoPE"
    try:
        torch.manual_seed(0)
        c = LTX2TextConnectors(video_connector_num_layers=8, audio_connector_num_layers=8, per_modality_projections=True,
                               video_connector_num_attention_heads=32, audio_connector_num_attention_heads=16,  # registers = heads x 128 must match 4096 / 2048
                               rope_double_precision=double).eval().requires_grad_(False)
        with torch.no_grad():
            if not double:
                ref = c(emb, mask, padding_side="left")
            c = c.to(dev, torch.bfloat16)
            t = time.perf_counter()
            out = c(emb.to(dev, torch.bfloat16), mask.to(dev), padding_side="left"); sync()
            dt = time.perf_counter() - t
        s = f"  {tag}: ran, {dt:.2f} s, video {tuple(out[0].shape)} audio {tuple(out[1].shape)} finite={all(bool(torch.isfinite(x).all()) for x in out[:2])}"
        if not double:
            s += f"  rel diff vs CPU fp32 (float64 RoPE): video {rel(out[0], ref[0]):.4f} audio {rel(out[1], ref[1]):.4f}"
        print(s)
        del c, out
    except Exception as e:
        print(f"  {tag}: FAILED {first_line(e)}")
