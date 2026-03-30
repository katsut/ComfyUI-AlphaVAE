import os
import json
import torch
import folder_paths
import comfy.model_management as model_management


class AlphaVAEModel:
    """Wrapper for AlphaVAE (diffusers AutoencoderKL with 4ch RGBA support)."""

    def __init__(self, model, device, dtype):
        self.model = model
        self.device = device
        self.dtype = dtype

    def encode(self, image_rgba):
        """Encode RGBA image [B,4,H,W] in [-1,1] to latent [B,16,H/8,W/8]."""
        self.model.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            posterior = self.model.encode(image_rgba.to(self.device, dtype=self.dtype))
            latent = posterior.latent_dist.sample()
        self.model.to(model_management.vae_offload_device())
        return latent

    def decode(self, latent):
        """Decode latent [B,16,H/8,W/8] to RGBA image [B,4,H,W] in [-1,1].

        Calls the decoder directly to bypass diffusers' auto-scaling,
        since ComfyUI's KSampler already handles latent scaling via process_out.
        """
        self.model.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            decoded = self.model.decoder(latent.to(self.device, dtype=self.dtype))
        self.model.to(model_management.vae_offload_device())
        return decoded


class AlphaVAELoader:
    """Load AlphaVAE model from a diffusers-format directory."""

    @classmethod
    def INPUT_TYPES(cls):
        vae_dir = os.path.join(folder_paths.models_dir, "vae")
        candidates = []
        if os.path.isdir(vae_dir):
            for name in os.listdir(vae_dir):
                name_path = os.path.join(vae_dir, name)
                if not os.path.isdir(name_path):
                    continue
                config_path = os.path.join(name_path, "config.json")
                if os.path.isfile(config_path):
                    candidates.append(name)
                else:
                    # Check one level deeper (e.g. AlphaVAE/finetune_VAE/)
                    for sub in os.listdir(name_path):
                        sub_path = os.path.join(name_path, sub)
                        if os.path.isdir(sub_path) and os.path.isfile(os.path.join(sub_path, "config.json")):
                            candidates.append(os.path.join(name, sub))
        if not candidates:
            candidates = ["(no diffusers VAE found)"]
        return {
            "required": {
                "vae_name": (candidates, {"tooltip": "Select AlphaVAE model directory (must contain config.json + diffusion_pytorch_model.safetensors)"}),
            }
        }

    RETURN_TYPES = ("ALPHA_VAE",)
    RETURN_NAMES = ("alpha_vae",)
    FUNCTION = "load_vae"
    CATEGORY = "AlphaVAE"

    def load_vae(self, vae_name):
        from diffusers import AutoencoderKL

        vae_path = os.path.join(folder_paths.models_dir, "vae", vae_name)
        config_path = os.path.join(vae_path, "config.json")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"AlphaVAE config not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        if config.get("in_channels") != 4 or config.get("out_channels") != 4:
            raise ValueError(
                f"Expected 4-channel VAE, got in_channels={config.get('in_channels')}, "
                f"out_channels={config.get('out_channels')}"
            )

        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device)

        vae = AutoencoderKL.from_pretrained(vae_path)
        vae.eval()
        vae.to(model_management.vae_offload_device())

        return (AlphaVAEModel(vae, device, dtype),)


class AlphaVAEDecode:
    """Decode latent to RGBA image using AlphaVAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "alpha_vae": ("ALPHA_VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "decode"
    CATEGORY = "AlphaVAE"

    def decode(self, samples, alpha_vae):
        latent = samples["samples"]

        # KSampler process_out already converts to VAE's natural latent space.
        # We call decoder directly (bypassing diffusers auto-scaling),
        # so no additional scaling needed.
        decoded = alpha_vae.decode(latent)

        # Convert from [-1,1] to [0,1]
        rgba = (decoded + 1.0) / 2.0
        rgba = rgba.clamp(0.0, 1.0)

        # [B,4,H,W] -> [B,H,W,4] (ComfyUI IMAGE format)
        rgba = rgba.permute(0, 2, 3, 1).cpu().float()

        # Split RGB and Alpha
        image = rgba[:, :, :, :3]
        # AlphaVAE outputs inverted alpha (0=opaque, 1=transparent)
        alpha_mask = 1.0 - rgba[:, :, :, 3]

        return (image, alpha_mask)


class AlphaVAEEncode:
    """Encode RGBA image to latent using AlphaVAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
                "alpha_vae": ("ALPHA_VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "AlphaVAE"

    def encode(self, image, alpha, alpha_vae):
        # image: [B,H,W,3], alpha: [B,H,W]
        # Combine to RGBA
        if alpha.ndim == 3:
            alpha = alpha.unsqueeze(-1)  # [B,H,W,1]
        rgba = torch.cat([image, alpha], dim=-1)  # [B,H,W,4]

        # [B,H,W,4] -> [B,4,H,W]
        rgba = rgba.permute(0, 3, 1, 2)

        # Convert from [0,1] to [-1,1]
        rgba = rgba * 2.0 - 1.0

        # Encode to VAE's natural latent space
        # ComfyUI's KSampler will apply process_in (scaling) as needed
        latent = alpha_vae.encode(rgba)

        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "AlphaVAELoader": AlphaVAELoader,
    "AlphaVAEDecode": AlphaVAEDecode,
    "AlphaVAEEncode": AlphaVAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaVAELoader": "AlphaVAE Loader",
    "AlphaVAEDecode": "AlphaVAE Decode (RGBA)",
    "AlphaVAEEncode": "AlphaVAE Encode (RGBA)",
}
