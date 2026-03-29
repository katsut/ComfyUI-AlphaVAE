# ComfyUI-AlphaVAE

ComfyUI custom nodes for [AlphaVAE](https://github.com/o0o0o00o0/AlphaVAE) — native RGBA transparent image generation using FLUX.

AlphaVAE replaces the standard FLUX VAE with an RGBA-capable VAE, enabling direct transparent image generation without post-processing background removal.

## Nodes

| Node | Description |
|------|-------------|
| `AlphaVAE Loader` | Load AlphaVAE model (diffusers format) |
| `AlphaVAE Decode` | Decode latent to RGBA image (outputs IMAGE + MASK) |
| `AlphaVAE Encode` | Encode RGBA image to latent |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/katsut/ComfyUI-AlphaVAE.git
pip install diffusers>=0.33.0
```

## Required Models

### 1. FLUX.1-dev (base model)

Download `flux1-dev.safetensors` from [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) (requires license agreement).

Place in: `ComfyUI/models/diffusion_models/flux1-dev.safetensors`

### 2. AlphaVAE VAE (168 MB)

Download `finetune_VAE/` directory from [HuggingFace](https://huggingface.co/o0o0o00o0/AlphaVAE).

Place in: `ComfyUI/models/vae/AlphaVAE/finetune_VAE/` (must contain `config.json` + `diffusion_pytorch_model.safetensors`)

### 3. AlphaVAE Diffusion LoRA (1.3 GB) — Required

Download `finetune_VAE/finetune_diffusion/pytorch_lora_weights.safetensors` from [HuggingFace](https://huggingface.co/o0o0o00o0/AlphaVAE).

Place in: `ComfyUI/models/loras/` (rename to e.g. `alphavae_diffusion_lora.safetensors`)

**Important:** The LoRA is required for proper transparency generation. Without it, AlphaVAE produces flat alpha values with no useful transparency information.

### 4. CLIP models (for FLUX text encoding)

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

## License

Code: MIT License

### Model License Notice

AlphaVAE model weights are a fine-tuned derivative of FLUX.1-dev's VAE. FLUX.1-dev is released under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). As a derivative work, AlphaVAE weights may inherit FLUX.1-dev's non-commercial restrictions regardless of the license stated on the AlphaVAE repository. Commercial use may require a separate license from [Black Forest Labs](https://blackforestlabs.ai/).

This project distributes only wrapper code, not model weights. Users are responsible for obtaining model weights and complying with applicable licenses.

## References

- [AlphaVAE Paper (arXiv:2507.09308)](https://arxiv.org/abs/2507.09308)
- [AlphaVAE GitHub](https://github.com/o0o0o00o0/AlphaVAE)
- [AlphaVAE HuggingFace](https://huggingface.co/o0o0o00o0/AlphaVAE)
- [LayerDiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse) — Prior art (SDXL, unmaintained)
