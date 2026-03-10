"""
Seamless Latent Tiling - ComfyUI Custom Node
=============================================

Installation:
  Copy this folder into ComfyUI/custom_nodes/seamless_latent_tiling/

Usage:
  Connect between your model loader and KSampler:
    Model Loader → [Seamless Latent Tiling ✦] → KSampler

  The node circular-pads the latent at every denoising step so that
  the UNet sees wrapped edges, producing perfectly tileable images.

Parameters:
  • padding   – overlap in image pixels (÷8 internally for latent space).
                64–128 is a good default. More = better awareness, slower.
  • tile_x    – tile horizontally (left ↔ right)
  • tile_y    – tile vertically  (top ↔ bottom)

Limitations:
  • ControlNet spatial patches are NOT padded — ControlNet may break
    seamlessness at the edges. For best results, avoid ControlNet or
    use only non-spatial guidance (e.g. IP-Adapter style conditioning).
  • Memory increases proportionally with padding size because the UNet
    processes a larger tensor at each step.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
