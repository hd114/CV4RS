import torch
import numpy as np

def apply_pruning(model: torch.nn.Module, pruning_ratio: float = 0.8) -> dict:
    """
    Apply structured pruning to the model by setting a fraction of parameters to zero.

    Args:
        model (torch.nn.Module): The global model to be pruned.
        pruning_ratio (float): Fraction of parameters to prune (set to zero).

    Returns:
        dict: A dictionary containing the pruning mask and pruned state_dict.
    """
    pruned_state_dict = {}
    pruning_mask = {}

    for name, param in model.state_dict().items():
        # Skip non-floating-point tensors (e.g., running_mean, running_var)
        if not isinstance(param, torch.Tensor) or not param.is_floating_point():
            pruned_state_dict[name] = param
            continue

        # Flatten the tensor to create a mask
        param_flat = param.view(-1)
        num_weights = param_flat.numel()
        num_pruned = int(pruning_ratio * num_weights)

        # Create indices to prune (set to 0)
        prune_indices = torch.randperm(num_weights)[:num_pruned]

        # Initialize mask with ones and set prune indices to zero
        mask = torch.ones_like(param_flat)
        mask[prune_indices] = 0
        mask = mask.view_as(param)  # Reshape back to original parameter shape

        pruning_mask[name] = mask
        pruned_state_dict[name] = param * mask  # Apply mask to parameters

        # Log the pruning details
        print(f"Layer: {name} | Mask shape: {mask.shape} | Pruned: {num_pruned}/{num_weights} ({pruning_ratio:.2f})")

    return {"pruned_state_dict": pruned_state_dict, "pruning_mask": pruning_mask}
