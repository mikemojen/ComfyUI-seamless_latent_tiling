# Seamless Latent Tiling — ComfyUI Custom Node

Generates perfectly tileable/repeating patterns by circular-padding the latent tensor at every denoising step.

## How It Works

```
Normal generation:          With this node:

 ┌──────────┐               ┌──┬──────────┬──┐
 │          │               │ R│          │ L│  ← right edge wraps to left
 │  Latent  │    ──►        │  │  Latent  │  │
 │          │               │ R│          │ L│
 └──────────┘               └──┴──────────┴──┘
                                    ↑
  Edges are independent       UNet sees wrapped context,
  → seams when tiled          so predictions at edges account
                              for what's on the other side
                              → seamless tiling
```

At each denoising step:
1. **Pad** the latent with circular copies (right edge wraps to left, bottom to top)
2. **Run** the UNet on this larger wrapped tensor
3. **Crop** the output back to the original size
4. The sampler continues as normal

The result tiles perfectly because the model "sees" across the seam boundaries during generation.

## Installation

Copy the entire `seamless_latent_tiling` folder into:
```
ComfyUI/custom_nodes/seamless_latent_tiling/
```
Restart ComfyUI. The node appears under **sampling/seamless**.

## Workflow

```
Model Loader  →  [Seamless Latent Tiling ✦]  →  KSampler  →  VAE Decode  →  Save
                        ↑
                   Connect MODEL
                   output/input
```

Simply insert the node between your model loader and sampler. Everything else stays the same.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **padding** | 64 | Overlap in image-space pixels (divided by 8 internally for latent space). Controls how much context the UNet sees across edges. |
| **tile_x** | True | Enable horizontal seamless tiling (left ↔ right) |
| **tile_y** | True | Enable vertical seamless tiling (top ↔ bottom) |

### Padding Guidelines

- **64 px** (8 latent px) — Fast, works for simple patterns
- **128 px** (16 latent px) — Good balance, recommended starting point
- **256 px** (32 latent px) — Maximum context, best for complex scenes, slower

## Tips for Best Results

1. **Use pattern-oriented prompts**: Include words like "seamless pattern", "tileable texture", "repeating design" in your prompt.

2. **Square images work best**: Use 512×512 or 768×768 for most reliable tiling.

3. **Increase padding if you see faint seams**: If the edges almost-but-not-quite match, increase the padding value.

4. **tile_x / tile_y independently**: If you only need horizontal tiling (e.g., a border strip), disable tile_y to save memory and get better vertical composition.

5. **Verify your result**: After generating, tile the image 2×2 or 3×3 in any image editor to visually confirm seamlessness.

## Known Limitations

- **ControlNet**: ControlNet injects spatial features at intermediate UNet layers that are NOT padded by this node. ControlNet guidance may break seamlessness near edges. Use without ControlNet for guaranteed results.

- **Memory overhead**: The UNet processes a slightly larger tensor. With 128px padding on a 512×512 image, the latent goes from 64×64 to 96×96 (~2.25× area). Plan GPU memory accordingly.

- **Inpainting models**: Concat conditioning (c_concat) IS padded when it matches the latent spatial dimensions, so inpaint models should work. However, this is less tested.

## Compatibility

- Works with SD 1.5, SDXL, and turbo/lightning variants (including z-image-turbo)
- Works with any sampler/scheduler in ComfyUI
- Works with LoRAs and textual inversions
- Does NOT require model modifications — purely a sampling-time wrapper
