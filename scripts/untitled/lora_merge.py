import torch
import safetensors.torch
from safetensors import safe_open
import scripts.untitled.common as cmn
import scripts.untitled.misc_util as mutil
from modules import sd_models
import re


def get_lora_keys(lora_path):
    """Get all keys from a LoRA file"""
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        return list(f.keys())


def parse_lora_key(lora_key):
    """Parse LoRA key to extract base model key and LoRA type (up/down/alpha)"""
    # LoRA keys typically look like:
    # lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight
    # lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight

    # Extract the lora type (up, down, alpha)
    if '.lora_down.weight' in lora_key:
        lora_type = 'down'
        base_key = lora_key.replace('.lora_down.weight', '.weight')
    elif '.lora_up.weight' in lora_key:
        lora_type = 'up'
        base_key = lora_key.replace('.lora_up.weight', '.weight')
    elif '.alpha' in lora_key:
        lora_type = 'alpha'
        base_key = lora_key.replace('.alpha', '.weight')
    else:
        return None, None

    # Convert LoRA naming to model naming
    # lora_unet_ -> model.diffusion_model.
    # lora_te_text_model_ -> cond_stage_model.transformer.text_model.

    if base_key.startswith('lora_unet_'):
        base_key = base_key.replace('lora_unet_', 'model.diffusion_model.', 1)
    elif base_key.startswith('lora_te_text_model_') or base_key.startswith('lora_te1_text_model_'):
        base_key = base_key.replace('lora_te_text_model_', 'cond_stage_model.transformer.text_model.', 1)
        base_key = base_key.replace('lora_te1_text_model_', 'cond_stage_model.transformer.text_model.', 1)
    elif base_key.startswith('lora_te2_text_model_'):
        # SDXL second text encoder
        base_key = base_key.replace('lora_te2_text_model_', 'conditioner.embedders.1.model.', 1)

    # Replace underscores with dots for standard pytorch naming
    # This is a simple heuristic and may need adjustment
    # e.g., down_blocks_0 -> down_blocks.0
    base_key = re.sub(r'_(\d+)_', r'.\1.', base_key)
    base_key = base_key.replace('_', '.')

    return base_key, lora_type


def merge_lora_to_checkpoint(checkpoint_path, lora_path, output_path, strength=1.0, progress=None):
    """
    Merge a LoRA into a checkpoint by applying the LoRA transformations

    Args:
        checkpoint_path: Path to base checkpoint
        lora_path: Path to LoRA file
        output_path: Where to save merged checkpoint
        strength: How strongly to apply the LoRA (0-1, can go higher)
        progress: Progress callback function
    """
    if progress:
        progress(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    with safe_open(checkpoint_path, framework='pt', device='cpu') as checkpoint_file:
        checkpoint_keys = list(checkpoint_file.keys())
        checkpoint_dict = {k: checkpoint_file.get_tensor(k) for k in checkpoint_keys}

    if progress:
        progress(f"Loading LoRA: {lora_path}")

    # Load LoRA
    with safe_open(lora_path, framework='pt', device='cpu') as lora_file:
        lora_keys = list(lora_file.keys())
        lora_dict = {k: lora_file.get_tensor(k) for k in lora_keys}

    if progress:
        progress(f"Merging LoRA into checkpoint...")

    # Group LoRA keys by their base key
    lora_groups = {}
    for lora_key in lora_keys:
        base_key, lora_type = parse_lora_key(lora_key)
        if base_key is None:
            continue

        if base_key not in lora_groups:
            lora_groups[base_key] = {}
        lora_groups[base_key][lora_type] = lora_key

    merged_count = 0
    skipped_count = 0

    # Apply LoRA to checkpoint
    for base_key, lora_parts in lora_groups.items():
        if 'up' not in lora_parts or 'down' not in lora_parts:
            continue

        # Try to find matching checkpoint key (may need fuzzy matching)
        checkpoint_key = None
        for ck_key in checkpoint_keys:
            if base_key in ck_key or ck_key in base_key:
                checkpoint_key = ck_key
                break

        if checkpoint_key is None or checkpoint_key not in checkpoint_dict:
            skipped_count += 1
            continue

        try:
            # Get LoRA components
            lora_up = lora_dict[lora_parts['up']]
            lora_down = lora_dict[lora_parts['down']]

            # Get alpha (default to rank if not present)
            if 'alpha' in lora_parts:
                alpha = lora_dict[lora_parts['alpha']].item()
            else:
                alpha = float(lora_down.shape[0])  # Use rank as default alpha

            # Calculate rank
            rank = lora_down.shape[0]

            # Calculate LoRA delta: (up @ down) * (alpha / rank) * strength
            # up shape: [out_dim, rank], down shape: [rank, in_dim]
            lora_delta = (lora_up @ lora_down) * (alpha / rank) * strength

            # Apply to checkpoint weight
            original_weight = checkpoint_dict[checkpoint_key]

            # Ensure shapes match
            if original_weight.shape == lora_delta.shape:
                checkpoint_dict[checkpoint_key] = original_weight + lora_delta
                merged_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            if progress:
                progress(f"Error merging {base_key}: {str(e)}")
            skipped_count += 1
            continue

    if progress:
        progress(f"Merged {merged_count} layers, skipped {skipped_count}")
        progress(f"Saving to: {output_path}")

    # Save merged checkpoint
    safetensors.torch.save_file(checkpoint_dict, output_path)

    if progress:
        progress(f"✓ LoRA merge complete!", popup=True)

    return f"Successfully merged {merged_count} layers (skipped {skipped_count})"


def merge_loras(lora_paths, output_path, weights=None, progress=None):
    """
    Merge multiple LoRA files together

    Args:
        lora_paths: List of LoRA file paths
        output_path: Where to save merged LoRA
        weights: List of weights for each LoRA (default: equal weights)
        progress: Progress callback function
    """
    if weights is None:
        weights = [1.0 / len(lora_paths)] * len(lora_paths)

    if len(lora_paths) != len(weights):
        raise ValueError("Number of LoRAs must match number of weights")

    if progress:
        progress(f"Merging {len(lora_paths)} LoRA files...")

    # Load all LoRAs
    all_loras = []
    for lora_path in lora_paths:
        with safe_open(lora_path, framework='pt', device='cpu') as lora_file:
            lora_keys = list(lora_file.keys())
            lora_dict = {k: lora_file.get_tensor(k) for k in lora_keys}
            all_loras.append(lora_dict)

    # Get union of all keys
    all_keys = set()
    for lora_dict in all_loras:
        all_keys.update(lora_dict.keys())

    # Merge LoRAs
    merged_dict = {}
    for key in all_keys:
        # Sum weighted tensors
        merged_tensor = None
        for lora_dict, weight in zip(all_loras, weights):
            if key in lora_dict:
                tensor = lora_dict[key] * weight
                if merged_tensor is None:
                    merged_tensor = tensor
                else:
                    merged_tensor = merged_tensor + tensor

        if merged_tensor is not None:
            merged_dict[key] = merged_tensor

    if progress:
        progress(f"Saving merged LoRA to: {output_path}")

    # Save merged LoRA
    safetensors.torch.save_file(merged_dict, output_path)

    if progress:
        progress(f"✓ LoRA merge complete!", popup=True)

    return f"Successfully merged {len(lora_paths)} LoRAs into {len(merged_dict)} keys"
