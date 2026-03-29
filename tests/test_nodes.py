"""Unit tests for ComfyUI-AlphaVAE nodes."""

import importlib.util
import json
import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

# folder_paths and comfy are mocked in conftest.py
import folder_paths


@pytest.fixture
def nodes_module():
    """Import nodes module fresh for each test."""
    for mod_name in list(sys.modules):
        if mod_name == "nodes":
            del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        "nodes", os.path.join(os.path.dirname(__file__), "..", "nodes.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def vae_dir(tmp_path):
    """Set up a temporary models/vae directory."""
    vae_path = tmp_path / "models" / "vae"
    vae_path.mkdir(parents=True)
    folder_paths.models_dir = str(tmp_path / "models")
    return vae_path


class TestNodeRegistration:
    def test_node_class_mappings_has_all_nodes(self, nodes_module):
        mappings = nodes_module.NODE_CLASS_MAPPINGS
        assert "AlphaVAELoader" in mappings
        assert "AlphaVAEDecode" in mappings
        assert "AlphaVAEEncode" in mappings

    def test_node_display_name_mappings_has_all_nodes(self, nodes_module):
        mappings = nodes_module.NODE_DISPLAY_NAME_MAPPINGS
        assert "AlphaVAELoader" in mappings
        assert "AlphaVAEDecode" in mappings
        assert "AlphaVAEEncode" in mappings


class TestAlphaVAELoader:
    def test_input_types_structure(self, nodes_module, vae_dir):
        input_types = nodes_module.AlphaVAELoader.INPUT_TYPES()
        assert "required" in input_types
        assert "vae_name" in input_types["required"]

    def test_return_types(self, nodes_module):
        assert nodes_module.AlphaVAELoader.RETURN_TYPES == ("ALPHA_VAE",)

    def test_category(self, nodes_module):
        assert nodes_module.AlphaVAELoader.CATEGORY == "AlphaVAE"

    def test_discovers_diffusers_vae_directory(self, nodes_module, vae_dir):
        test_vae = vae_dir / "TestVAE"
        test_vae.mkdir()
        (test_vae / "config.json").write_text(
            json.dumps({"in_channels": 4, "out_channels": 4})
        )

        input_types = nodes_module.AlphaVAELoader.INPUT_TYPES()
        candidates = input_types["required"]["vae_name"][0]
        assert "TestVAE" in candidates

    def test_discovers_nested_vae_directory(self, nodes_module, vae_dir):
        nested = vae_dir / "AlphaVAE" / "finetune_VAE"
        nested.mkdir(parents=True)
        (nested / "config.json").write_text(
            json.dumps({"in_channels": 4, "out_channels": 4})
        )

        input_types = nodes_module.AlphaVAELoader.INPUT_TYPES()
        candidates = input_types["required"]["vae_name"][0]
        assert os.path.join("AlphaVAE", "finetune_VAE") in candidates

    def test_no_vae_shows_placeholder(self, nodes_module, vae_dir):
        input_types = nodes_module.AlphaVAELoader.INPUT_TYPES()
        candidates = input_types["required"]["vae_name"][0]
        assert candidates == ["(no diffusers VAE found)"]

    def test_rejects_non_4channel_vae(self, nodes_module, vae_dir):
        wrong_vae = vae_dir / "WrongVAE"
        wrong_vae.mkdir()
        (wrong_vae / "config.json").write_text(
            json.dumps({"in_channels": 3, "out_channels": 3})
        )

        with pytest.raises(ValueError, match="Expected 4-channel VAE"):
            nodes_module.AlphaVAELoader().load_vae("WrongVAE")


class TestAlphaVAEDecode:
    def test_input_types_structure(self, nodes_module):
        input_types = nodes_module.AlphaVAEDecode.INPUT_TYPES()
        assert "required" in input_types
        assert "samples" in input_types["required"]
        assert "alpha_vae" in input_types["required"]

    def test_return_types(self, nodes_module):
        assert nodes_module.AlphaVAEDecode.RETURN_TYPES == ("IMAGE", "MASK")
        assert nodes_module.AlphaVAEDecode.RETURN_NAMES == ("image", "alpha")

    def test_decode_output_shapes(self, nodes_module):
        mock_vae = MagicMock()
        mock_vae.decode.return_value = torch.randn(1, 4, 64, 64)

        samples = {"samples": torch.randn(1, 16, 8, 8)}
        image, alpha = nodes_module.AlphaVAEDecode().decode(samples, mock_vae)

        assert image.shape == (1, 64, 64, 3)
        assert alpha.shape == (1, 64, 64)

    def test_decode_output_range(self, nodes_module):
        mock_vae = MagicMock()
        mock_vae.decode.return_value = torch.tensor(
            [
                [
                    [[2.0, -2.0], [0.5, -0.5]],
                    [[1.0, -1.0], [0.0, 0.0]],
                    [[0.5, 0.5], [-0.5, -0.5]],
                    [[1.0, -1.0], [0.0, 0.5]],
                ]
            ]
        )

        samples = {"samples": torch.randn(1, 16, 1, 1)}
        image, alpha = nodes_module.AlphaVAEDecode().decode(samples, mock_vae)

        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0


class TestAlphaVAEEncode:
    def test_input_types_structure(self, nodes_module):
        input_types = nodes_module.AlphaVAEEncode.INPUT_TYPES()
        assert "required" in input_types
        assert "image" in input_types["required"]
        assert "alpha" in input_types["required"]
        assert "alpha_vae" in input_types["required"]

    def test_return_types(self, nodes_module):
        assert nodes_module.AlphaVAEEncode.RETURN_TYPES == ("LATENT",)

    def test_encode_output_structure(self, nodes_module):
        mock_vae = MagicMock()
        mock_vae.encode.return_value = torch.randn(1, 16, 8, 8)

        image = torch.randn(1, 64, 64, 3)
        alpha = torch.randn(1, 64, 64)

        result = nodes_module.AlphaVAEEncode().encode(image, alpha, mock_vae)
        assert isinstance(result[0], dict)
        assert "samples" in result[0]
        assert result[0]["samples"].shape == (1, 16, 8, 8)

    def test_encode_combines_rgba(self, nodes_module):
        mock_vae = MagicMock()
        mock_vae.encode.return_value = torch.randn(1, 16, 8, 8)

        image = torch.ones(1, 64, 64, 3) * 0.5
        alpha = torch.ones(1, 64, 64) * 0.8

        nodes_module.AlphaVAEEncode().encode(image, alpha, mock_vae)

        call_args = mock_vae.encode.call_args[0][0]
        assert call_args.shape == (1, 4, 64, 64)
        assert torch.allclose(call_args[:, :3], torch.zeros(1, 3, 64, 64), atol=1e-6)
        assert torch.allclose(
            call_args[:, 3:], torch.full((1, 1, 64, 64), 0.6), atol=1e-6
        )


class TestAlphaVAEModel:
    def test_decode_calls_decoder_directly(self, nodes_module):
        mock_model = MagicMock()
        mock_model.decoder.return_value = torch.randn(1, 4, 64, 64)

        vae = nodes_module.AlphaVAEModel(
            mock_model, torch.device("cpu"), torch.float32
        )
        vae.decode(torch.randn(1, 16, 8, 8))

        mock_model.decoder.assert_called_once()
        mock_model.decode.assert_not_called()

    def test_encode_uses_model_encode(self, nodes_module):
        mock_model = MagicMock()
        mock_posterior = MagicMock()
        mock_posterior.latent_dist.sample.return_value = torch.randn(1, 16, 8, 8)
        mock_model.encode.return_value = mock_posterior

        vae = nodes_module.AlphaVAEModel(
            mock_model, torch.device("cpu"), torch.float32
        )
        vae.encode(torch.randn(1, 4, 64, 64))

        mock_model.encode.assert_called_once()

    def test_device_offload_after_decode(self, nodes_module):
        mock_model = MagicMock()
        mock_model.decoder.return_value = torch.randn(1, 4, 64, 64)

        vae = nodes_module.AlphaVAEModel(
            mock_model, torch.device("cpu"), torch.float32
        )
        vae.decode(torch.randn(1, 16, 8, 8))

        assert mock_model.to.call_count == 2
