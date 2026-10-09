"""Accelerator selection shared by every script: CUDA (NVIDIA) if available, else XPU (Intel Arc), else MPS (Apple).
Override with LTX_DEVICE=cuda|xpu|mps. `acc` is torch.cuda or torch.xpu (same API for synchronize/empty_cache/Stream/stream),
or a shim with the same calls for MPS. HAS_STREAMS is False on MPS: there are no side streams, so copies run on the main thread."""
import contextlib
import os
import types

import torch


def _pick():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "xpu"   # keep the old default so the error message names the expected backend


NAME = os.environ.get("LTX_DEVICE") or _pick()
DEV = torch.device(NAME)
HAS_STREAMS = NAME != "mps"
MODELS = os.environ.get("LTX_MODELS", "models/ltx25")   # folder holding the downloaded checkpoint

if NAME == "mps":
    class _NoStream:
        """MPS has one command queue per process; synchronize() on it is a device-wide sync."""
        def synchronize(self):
            torch.mps.synchronize()

    def _mem_get_info():
        used = torch.mps.driver_allocated_memory()
        return max(torch.mps.recommended_max_memory() - used, 0), torch.mps.recommended_max_memory()

    acc = types.SimpleNamespace(
        synchronize=torch.mps.synchronize,
        empty_cache=torch.mps.empty_cache,
        memory_allocated=torch.mps.current_allocated_memory,
        mem_get_info=_mem_get_info,
        Stream=_NoStream,
        stream=lambda s: contextlib.nullcontext(),
    )
else:
    acc = getattr(torch, NAME)


def _patch_rope_for_mps():
    """MPS has no float64. LTX-2 RoPE (rope_double_precision=True) only uses it for one pow() and casts to float32 right after,
    so run those forwards on the CPU and move the float32 cos/sin tables to the GPU: same numbers as the XPU path, negligible cost."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed
    from diffusers.pipelines.ltx2.connectors import LTX2RotaryPosEmbed1d

    for cls in (LTX2AudioVideoRotaryPosEmbed, LTX2RotaryPosEmbed1d):
        if getattr(cls, "_cpu_rope", False):
            continue
        orig = cls.forward

        def forward(self, *args, device=None, _orig=orig, **kw):
            if not self.double_precision:
                return _orig(self, *args, device=device, **kw)
            target = device or next(a.device for a in args if isinstance(a, torch.Tensor))
            args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
            return tuple(o.to(target) for o in _orig(self, *args, device="cpu", **kw))

        cls.forward, cls._cpu_rope = forward, True


if NAME == "mps":
    _patch_rope_for_mps()
