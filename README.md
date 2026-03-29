# ComfyUI-AlphaVAE

ComfyUI custom nodes for [AlphaVAE](https://github.com/o0o0o00o0/AlphaVAE) — native RGBA transparent image generation using FLUX.

AlphaVAE replaces the standard FLUX VAE with an RGBA-capable VAE, enabling direct transparent image generation without post-processing background removal.

## Status

🚧 **Work in Progress** — Not yet functional.

## Planned Nodes

| Node | Description |
|------|-------------|
| `AlphaVAE Loader` | Load AlphaVAE model (diffusers format) |
| `AlphaVAE Decode` | Decode latent to RGBA image (outputs IMAGE + MASK) |
| `AlphaVAE Encode` | Encode RGBA image to latent |

## Requirements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) v0.18+
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) base model
- [AlphaVAE](https://huggingface.co/o0o0o00o0/AlphaVAE) model weights

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
