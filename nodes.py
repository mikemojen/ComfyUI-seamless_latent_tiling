"""
Seamless Latent Tiling Node for ComfyUI
========================================

Patches the model's UNet forward pass to use circular (wrap-around) padding
on the latent tensor at each denoising step. This makes the diffusion process
"aware" that the left edge connects to the right edge (and top to bottom),
producing images that tile perfectly by construction.

How it works:
  1. Before each UNet evaluation, the noisy latent is padded with circular
     copies of itself (the right edge wraps to the left, bottom wraps to top).
  2. The UNet runs on this larger, wrapped tensor — so every convolution
     at every layer sees continuous context across the seam boundaries.
  3. The output noise prediction is cropped back to the original size.
  4. The sampler/scheduler proceeds normally with the cropped prediction.

The result is a latent (and decoded image) whose edges match perfectly
when placed side-by-side.

Usage in ComfyUI:
  MODEL_LOADER  →  [Seamless Latent Tiling]  →  SAMPLER
  Connect this node between your model loader and your KSampler.

Seamless Post-Process Nodes for ComfyUI
========================================

Two-stage post-processing to eliminate remaining seams after generation:

Stage 1 — SeamlessColorHarmonize:
    Measures color/brightness differences between opposing edge strips
    and applies a smooth correction gradient across the image to equalize
    them. Fixes the subtle color banding visible at seams.

Stage 2 — SeamlessOffsetForInpaint + SeamlessOffsetReverse:
    Shifts the image by 50% so seams move to the center, generates a
    feathered cross-shaped mask over the seam area, and outputs both
    for use with any inpaint sampler. After inpainting, the reverse
    node shifts back to restore original positioning.

Recommended workflow:
    IMAGE → [Color Harmonize] → [VAE Encode] → [Offset For Inpaint]
          → KSampler (inpaint) → [VAE Decode] → [Offset Reverse] → Save

All nodes handle arbitrary aspect ratios.
"""

import torch
import torch.nn.functional as F
import math

# ═══════════════════════════════════════════════════════════════════════════
# NODE 1: Color Harmonize
# ═══════════════════════════════════════════════════════════════════════════

class SeamlessColorHarmonize:
    """
    Equalizes color and brightness across opposing edges so that
    left↔right and top↔bottom transitions are smooth when tiled.

    Algorithm:
      1. Sample a strip of pixels along each edge.
      2. Compute the per-channel mean difference between opposing edges.
      3. Build a smooth (cosine) correction gradient across the full
         image that ramps from +half_diff at one edge to -half_diff
         at the opposite edge.
      4. Blend the correction into the image at the given strength.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strip_width": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 128,
                    "step": 2,
                    "tooltip": (
                        "Width of the edge strip (in pixels) used to measure "
                        "the color difference. Larger = more stable measurement "
                        "but averages over more of the image."
                    ),
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": (
                        "Correction strength. 1.0 = fully equalize the measured "
                        "difference. Values below 1 apply a partial correction. "
                        "Values above 1 overcorrect (useful if strips underestimate "
                        "the true edge difference)."
                    ),
                }),
                "harmonize_x": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Correct left↔right color difference.",
                }),
                "harmonize_y": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Correct top↔bottom color difference.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "harmonize"
    CATEGORY = "image/seamless"
    DESCRIPTION = (
        "Smoothly corrects color/brightness differences between opposing "
        "edges to eliminate visible seams when tiling."
    )

    def harmonize(self, image, strip_width, strength, harmonize_x, harmonize_y):
        if not harmonize_x and not harmonize_y:
            return (image,)

        # IMAGE shape: [B, H, W, C], values 0–1
        img = image.clone().float()
        B, H, W, C = img.shape

        # Clamp strip width to avoid exceeding image dimensions
        sw_x = min(strip_width, W // 4)
        sw_y = min(strip_width, H // 4)

        if harmonize_x and W > 1:
            # Mean color of left strip vs right strip
            left_mean = img[:, :, :sw_x, :].mean(dim=(1, 2), keepdim=True)   # [B,1,1,C]
            right_mean = img[:, :, -sw_x:, :].mean(dim=(1, 2), keepdim=True)  # [B,1,1,C]

            # For seamless tiling, the right edge of the image must match
            # the left edge. The difference to correct:
            diff = left_mean - right_mean  # [B, 1, 1, C]

            # Build a cosine ramp from +diff/2 (at right edge) to -diff/2 (at left edge)
            # When tiled: ...right_edge | left_edge... the correction makes them meet.
            #
            # ramp goes from -0.5 at x=0 to +0.5 at x=W-1
            t = torch.linspace(0.0, 1.0, W, device=img.device)
            # Cosine interpolation for smoother transition than linear
            ramp = 0.5 * (1.0 - torch.cos(t * math.pi)) - 0.5  # range: -0.5 to +0.5
            ramp = ramp.view(1, 1, W, 1)  # [1, 1, W, 1]

            correction_x = diff * ramp * strength  # [B, 1, W, C]
            img = img + correction_x

        if harmonize_y and H > 1:
            # Mean color of top strip vs bottom strip
            top_mean = img[:, :sw_y, :, :].mean(dim=(1, 2), keepdim=True)
            bottom_mean = img[:, -sw_y:, :, :].mean(dim=(1, 2), keepdim=True)

            diff = top_mean - bottom_mean

            t = torch.linspace(0.0, 1.0, H, device=img.device)
            ramp = 0.5 * (1.0 - torch.cos(t * math.pi)) - 0.5
            ramp = ramp.view(1, H, 1, 1)

            correction_y = diff * ramp * strength
            img = img + correction_y

        # Clamp to valid range
        img = img.clamp(0.0, 1.0)

        return (img,)


# ═══════════════════════════════════════════════════════════════════════════
# NODE 2: Offset For Inpaint
# ═══════════════════════════════════════════════════════════════════════════

class SeamlessOffsetForInpaint:
    """
    Shifts the image by 50% on each axis so seams move to the center,
    then generates a feathered cross-shaped mask over the seam region.

    Output the shifted image and mask to any inpaint workflow.
    After inpainting, use SeamlessOffsetReverse to shift back.

    The mask is a soft cross: full opacity (1.0) at the seam line,
    feathering to 0.0 at the mask edges. The feather_width controls
    the transition zone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_width": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": (
                        "Total width of the inpaint mask band in pixels. "
                        "Should be wide enough to cover the visible seam "
                        "plus blending room. 48–96 is typical."
                    ),
                }),
                "feather": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 256,
                    "step": 4,
                    "tooltip": (
                        "Soft falloff zone in pixels on each side of the mask band. "
                        "0 = hard-edged mask. Higher values = smoother blend into "
                        "surrounding content. Should be ≤ mask_width."
                    ),
                }),
                "offset_x": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Shift horizontally and mask the vertical seam.",
                }),
                "offset_y": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Shift vertically and mask the horizontal seam.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("shifted_image", "seam_mask", "shift_y", "shift_x")
    FUNCTION = "offset_and_mask"
    CATEGORY = "image/seamless"
    DESCRIPTION = (
        "Shifts image by 50% so seams move to center, then generates a "
        "feathered cross mask for inpainting. Use with SeamlessOffsetReverse "
        "after inpainting to shift back."
    )

    def offset_and_mask(self, image, mask_width, feather, offset_x, offset_y):
        # IMAGE: [B, H, W, C]
        B, H, W, C = image.shape

        # Compute shift amounts (half of each dimension)
        shift_y = H // 2 if offset_y else 0
        shift_x = W // 2 if offset_x else 0

        # Shift the image using torch.roll
        shifted = image.clone()
        if shift_y != 0:
            shifted = torch.roll(shifted, shifts=shift_y, dims=1)
        if shift_x != 0:
            shifted = torch.roll(shifted, shifts=shift_x, dims=2)

        # ── Build the feathered cross mask ────────────────────────
        # Mask = 1.0 where we want to inpaint (the seam), 0.0 elsewhere
        mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)

        # Effective feather cannot exceed mask_width
        eff_feather = min(feather, mask_width)
        # Half-width of the fully opaque region
        half_solid = max(0, (mask_width - eff_feather)) // 2
        # Half-width of the total mask region (solid + feather)
        half_total = half_solid + eff_feather

        if offset_y and H > 1:
            # Horizontal band across the center (at y = shift_y)
            cy = shift_y
            for dy in range(-half_total, half_total + 1):
                y = (cy + dy) % H
                dist_from_center = abs(dy)
                if dist_from_center <= half_solid:
                    val = 1.0
                elif eff_feather > 0:
                    # Cosine feather for smooth falloff
                    t = (dist_from_center - half_solid) / eff_feather
                    t = min(t, 1.0)
                    val = 0.5 * (1.0 + math.cos(t * math.pi))
                else:
                    val = 0.0
                mask[:, y, :] = torch.maximum(mask[:, y, :],
                                               torch.tensor(val, device=mask.device))

        if offset_x and W > 1:
            # Vertical band across the center (at x = shift_x)
            cx = shift_x
            for dx in range(-half_total, half_total + 1):
                x = (cx + dx) % W
                dist_from_center = abs(dx)
                if dist_from_center <= half_solid:
                    val = 1.0
                elif eff_feather > 0:
                    t = (dist_from_center - half_solid) / eff_feather
                    t = min(t, 1.0)
                    val = 0.5 * (1.0 + math.cos(t * math.pi))
                else:
                    val = 0.0
                mask[:, :, x] = torch.maximum(mask[:, :, x],
                                               torch.tensor(val, device=mask.device))

        return (shifted, mask, shift_y, shift_x)


# ═══════════════════════════════════════════════════════════════════════════
# NODE 3: Offset Reverse
# ═══════════════════════════════════════════════════════════════════════════

class SeamlessOffsetReverse:
    """
    Reverses the 50% shift applied by SeamlessOffsetForInpaint.
    Connect this after your inpaint → VAE decode to restore the
    original image positioning.

    Takes the shift_y and shift_x values output by SeamlessOffsetForInpaint
    so the reverse is always perfectly matched.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shift_y": ("INT", {
                    "default": 0,
                    "forceInput": True,
                    "tooltip": "Connect from SeamlessOffsetForInpaint shift_y output.",
                }),
                "shift_x": ("INT", {
                    "default": 0,
                    "forceInput": True,
                    "tooltip": "Connect from SeamlessOffsetForInpaint shift_x output.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "reverse_offset"
    CATEGORY = "image/seamless"
    DESCRIPTION = (
        "Reverses the shift from SeamlessOffsetForInpaint. "
        "Connect shift_y and shift_x from the offset node's outputs."
    )

    def reverse_offset(self, image, shift_y, shift_x):
        result = image.clone()

        # Reverse = shift by the negative amount
        if shift_y != 0:
            result = torch.roll(result, shifts=-shift_y, dims=1)
        if shift_x != 0:
            result = torch.roll(result, shifts=-shift_x, dims=2)

        return (result,)
    
# ═══════════════════════════════════════════════════════════════════════════
# NODE 4: Seamless Latent Tiling
# ═══════════════════════════════════════════════════════════════════════════

class SeamlessLatentTiling:
    """
    Wraps a model so that every denoising step uses circular-padded latents,
    producing seamlessly tileable images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "padding": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": (
                        "Overlap region in image-space pixels. "
                        "Internally divided by 8 for latent space. "
                        "Larger = better seam awareness but slower. "
                        "64-128 is a good starting range."
                    ),
                }),
                "tile_x": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable horizontal (left↔right) seamless tiling.",
                }),
                "tile_y": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable vertical (top↔bottom) seamless tiling.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_seamless_tiling"
    CATEGORY = "sampling/seamless"
    DESCRIPTION = (
        "Makes generated images seamlessly tileable by circular-padding "
        "the latent at every denoising step. Place between your model "
        "loader and sampler."
    )

    def apply_seamless_tiling(self, model, padding, tile_x, tile_y):
        # Nothing to do if both axes are disabled
        if not tile_x and not tile_y:
            return (model,)

        # Clone so we don't mutate the original model
        model_clone = model.clone()

        # Convert image-space pixels → latent-space pixels
        # Standard SD / SDXL VAE downscales by 8x
        latent_pad = max(1, padding // 8)

        # Capture in closure
        _tile_x = tile_x
        _tile_y = tile_y
        _latent_pad = latent_pad

        def seamless_unet_wrapper(apply_model_fn, params):
            """
            Wraps every call to the UNet's apply_model:
              1. Circular-pad the input latent
              2. Circular-pad any spatial conditioning tensors
              3. Run the UNet
              4. Crop the output back to the original size
            """
            x = params["input"]
            t = params["timestep"]
            c = params["c"]

            _, _, h, w = x.shape

            # Clamp padding to at most half the spatial dim to avoid
            # degenerate wrapping (where padding > tensor size)
            pad_h = min(_latent_pad, h // 2) if _tile_y else 0
            pad_w = min(_latent_pad, w // 2) if _tile_x else 0

            # Fast path: no padding needed
            if pad_h == 0 and pad_w == 0:
                return apply_model_fn(x, t, **c)

            # ── Pad the noisy latent ──────────────────────────────
            # F.pad order: (left, right, top, bottom) for 4-D tensors
            padded_x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="circular")

            # ── Pad spatial conditioning tensors ──────────────────
            # Text embeddings (c_crossattn) are non-spatial and pass through.
            # Concat conditioning (c_concat, used by inpaint models) IS
            # spatial and must be padded to match.
            padded_c = {}
            for key, value in c.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 4
                    and value.shape[2] == h
                    and value.shape[3] == w
                ):
                    padded_c[key] = F.pad(
                        value,
                        (pad_w, pad_w, pad_h, pad_h),
                        mode="circular",
                    )
                else:
                    padded_c[key] = value

            # ── Run the UNet on the padded tensor ─────────────────
            output = apply_model_fn(padded_x, t, **padded_c)

            # ── Crop back to original spatial dimensions ──────────
            output = output[:, :, pad_h : pad_h + h, pad_w : pad_w + w]

            return output

        model_clone.set_model_unet_function_wrapper(seamless_unet_wrapper)

        return (model_clone,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SeamlessLatentTiling": SeamlessLatentTiling,
    "SeamlessColorHarmonize": SeamlessColorHarmonize,
    "SeamlessOffsetForInpaint": SeamlessOffsetForInpaint,
    "SeamlessOffsetReverse": SeamlessOffsetReverse,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamlessLatentTiling": "Seamless Latent Tiling ✦",
    "SeamlessColorHarmonize": "Seamless Color Harmonize ✦",
    "SeamlessOffsetForInpaint": "Seamless Offset For Inpaint ✦",
    "SeamlessOffsetReverse": "Seamless Offset Reverse ✦",
}
