import torch
import matplotlib.pyplot as plt
import numpy as np


def apply_pruning(model: torch.nn.Module, pruning_ratio: float = 0.8, visualize: bool = False) -> dict:
    """
    Apply random pruning to the model by setting a fraction of parameters to zero.

    Args:
        model (torch.nn.Module): The global model to be pruned.
        pruning_ratio (float): Fraction of parameters to prune (set to zero).
        visualize (bool): If True, plots or prints the pruning mask for each layer.

    Returns:
        dict: A dictionary containing the pruning mask and pruned state_dict.
    """
    pruned_state_dict = {}
    pruning_mask = {}

    for name, param in model.state_dict().items():
        if param.requires_grad:
            # Create a random pruning mask
            mask = torch.rand_like(param) > pruning_ratio
            pruning_mask[name] = mask  # Save the mask for this layer

            # Apply mask to the parameters
            pruned_state_dict[name] = param * mask

            # Visualization
            if visualize:
                print(f"Layer: {name}")
                print(f"Mask shape: {mask.shape}")
                # Plotting only for smaller dimensions
                if len(mask.shape) <= 2 or (len(mask.shape) == 4 and mask.shape[0] <= 10):
                    # Convert the masked parameters to a NumPy array for visualization
                    param_np = param.cpu().numpy()
                    masked_param_np = param_np * mask.cpu().numpy()

                    # Create the color map
                    # Black for pruned neurons, blue for positive values, purple for negative values
                    visual_data = np.where(masked_param_np == 0, 0, masked_param_np)
                    cmap = plt.cm.coolwarm  # "coolwarm" transitions from blue to purple
                    norm = plt.Normalize(vmin=-np.max(np.abs(visual_data)), vmax=np.max(np.abs(visual_data)))

                    plt.imshow(visual_data, cmap=cmap, norm=norm, aspect="auto")
                    plt.colorbar(label="Value")
                    plt.title(f"Pruning Visualization for {name}")
                    plt.show()
                else:
                    print(f"Skipping visualization for large tensors: {mask.shape}")
        else:
            # If the parameter does not require gradient, keep it as is
            pruned_state_dict[name] = param

    return {"pruned_state_dict": pruned_state_dict, "pruning_mask": pruning_mask}
