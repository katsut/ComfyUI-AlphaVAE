# ComfyUI-AlphaVAE

ComfyUI custom nodes for [AlphaVAE](https://github.com/o0o0o00o0/AlphaVAE) — native RGBA transparent image generation using FLUX.

## What is this?

When creating game assets, stickers, or logos with AI image generation, you typically need **transparent backgrounds**. The standard approach is a two-step process: generate an image, then remove the background with a separate tool. This is slow and often produces rough edges.

**AlphaVAE** solves this by replacing the standard VAE (the component that converts between pixel images and the AI's internal representation) with one that natively understands transparency. Instead of generating RGB (3 channels), it generates RGBA (4 channels, where A = alpha/transparency) in a single pass.

This project provides the [ComfyUI](https://github.com/Comfy-Org/ComfyUI) integration nodes so you can use AlphaVAE in your workflows.

### How it fits together

```
┌─────────────────────────────────────────────────────┐
│ FLUX.1-dev          — Base diffusion model (by BFL) │
│   + AlphaVAE LoRA   — Teaches FLUX to "think" in   │
│                       transparency                  │
│   + AlphaVAE VAE    — Decodes latents to RGBA       │
│                       instead of RGB                │
│                                                     │
│ ComfyUI             — Node-based workflow engine    │
│   + ComfyUI-AlphaVAE — This project: the glue      │
│                        that connects AlphaVAE       │
│                        to ComfyUI                   │
└─────────────────────────────────────────────────────┘
```

### Why not LayerDiffuse?

[LayerDiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse) was the previous solution for native transparent image generation. However:

- It is **unmaintained** (last commit: Feb 2025) and incompatible with ComfyUI V3
- It only supports **SDXL/SD1.5** (no FLUX support)
- AlphaVAE achieves **better quality** (PSNR +4.9 dB, SSIM +3.2% vs LayerDiffuse)

## Nodes

| Node | Description |
|------|-------------|
| `AlphaVAE Loader` | Load AlphaVAE model (diffusers AutoencoderKL format) |
| `AlphaVAE Decode (RGBA)` | Decode latent → RGBA image (outputs IMAGE + MASK) |
| `AlphaVAE Encode (RGBA)` | Encode RGBA image → latent (IMAGE + MASK as inputs) |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/katsut/ComfyUI-AlphaVAE.git
pip install diffusers>=0.33.0
```

## Required Models

You need 4 components. None are bundled — download them yourself.

### 1. FLUX.1-dev (base diffusion model, ~12 GB)

The core image generation model by [Black Forest Labs](https://blackforestlabs.ai/).

Download `flux1-dev.safetensors` from [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) (requires license agreement).

Place in: `ComfyUI/models/diffusion_models/flux1-dev.safetensors`

### 2. AlphaVAE VAE (168 MB)

The modified VAE that outputs 4 channels (RGBA) instead of 3 (RGB). This is the core of AlphaVAE — a fine-tuned version of FLUX's VAE with expanded input/output channels.

Download `finetune_VAE/` directory from [HuggingFace](https://huggingface.co/o0o0o00o0/AlphaVAE).

Place in: `ComfyUI/models/vae/AlphaVAE/finetune_VAE/` (must contain `config.json` + `diffusion_pytorch_model.safetensors`)

### 3. AlphaVAE Diffusion LoRA (1.3 GB) — Required

A LoRA adapter that teaches FLUX to generate latents that encode transparency information. **Without this LoRA, FLUX has no concept of transparency**, and the alpha channel will be meaningless (flat values around 0.5).

Download `finetune_VAE/finetune_diffusion/pytorch_lora_weights.safetensors` from [HuggingFace](https://huggingface.co/o0o0o00o0/AlphaVAE).

Place in: `ComfyUI/models/loras/` (rename to e.g. `alphavae_diffusion_lora.safetensors`)

### 4. CLIP text encoders (for FLUX)

FLUX uses two text encoders to understand your prompts:

- `clip_l.safetensors` → `ComfyUI/models/clip/` ([download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors))
- `t5xxl_fp8_e4m3fn.safetensors` → `ComfyUI/models/clip/` ([download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors))

## Example Workflow

```
UNETLoader (flux1-dev.safetensors)
  → LoraLoader (alphavae_diffusion_lora.safetensors, strength: 1.0)
    → KSampler (steps: 20, cfg: 1.0, sampler: euler, scheduler: simple)

DualCLIPLoader (clip_l + t5xxl, type: flux)
  → LoraLoader (same)
    → CLIPTextEncode → FluxGuidance (guidance: 3.5)
      → KSampler

AlphaVAELoader (AlphaVAE/finetune_VAE)
  → AlphaVAEDecode (from KSampler output)
    → JoinImageWithAlpha (IMAGE + MASK → RGBA)
      → SaveImage
```

## Requirements

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) v0.18+
- [diffusers](https://github.com/huggingface/diffusers) >= 0.33.0
- ~36 GB disk space (FLUX ~12 GB + AlphaVAE ~1.5 GB + CLIP ~5 GB + workspace)

## License

Code: MIT License

### Model License Notice

AlphaVAE model weights are a fine-tuned derivative of FLUX.1-dev's VAE. FLUX.1-dev is released under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). As a derivative work, AlphaVAE weights may inherit FLUX.1-dev's non-commercial restrictions regardless of the license stated on the AlphaVAE repository. Commercial use may require a separate license from [Black Forest Labs](https://blackforestlabs.ai/).

This project distributes only wrapper code, not model weights. Users are responsible for obtaining model weights and complying with applicable licenses.

See also: [License clarification request to AlphaVAE authors](https://github.com/o0o0o00o0/AlphaVAE/issues/11)

## References

- [AlphaVAE Paper (arXiv:2507.09308)](https://arxiv.org/abs/2507.09308) — "AlphaVAE: Learning to generate transparent images with RGBA VAE"
- [AlphaVAE GitHub](https://github.com/o0o0o00o0/AlphaVAE)
- [AlphaVAE HuggingFace](https://huggingface.co/o0o0o00o0/AlphaVAE)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) — Base model by Black Forest Labs
- [LayerDiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse) — Prior art for SDXL (unmaintained)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) — Node-based Stable Diffusion workflow engine
