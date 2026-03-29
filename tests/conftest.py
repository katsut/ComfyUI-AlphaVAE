"""Pre-mock ComfyUI modules before any imports happen."""
import sys
import types

# Mock ComfyUI-specific modules that don't exist outside ComfyUI
folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = "/tmp/comfyui_test_models"

comfy = types.ModuleType("comfy")
comfy_mm = types.ModuleType("comfy.model_management")
comfy_mm.vae_device = lambda: __import__("torch").device("cpu")
comfy_mm.vae_dtype = lambda device=None: __import__("torch").float32
comfy_mm.vae_offload_device = lambda: __import__("torch").device("cpu")
comfy.model_management = comfy_mm

sys.modules["folder_paths"] = folder_paths
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = comfy_mm

collect_ignore = ["__init__.py", "nodes.py"]
