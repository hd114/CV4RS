# alt, funktioniert

import torch
import numpy as np

def apply_pruning(
    model: torch.nn.Module,
    pruning_ratio: float = 0.8,
    verbose: bool = True,
    check_nan: bool = True,
    least_relevant_first: bool = False,
) -> dict:
    """
    Apply structured pruning to the model by setting a fraction of parameters to zero.

    Args:
        model (torch.nn.Module): The global model to be pruned.
        pruning_ratio (float): Fraction of parameters to prune (set to zero).
        verbose (bool): If True, prints information about the pruning process.
        check_nan (bool): If True, checks for NaN values in pruned parameters.
        least_relevant_first (bool): If True, prunes the smallest-magnitude weights first.

    Returns:
        dict: A dictionary containing the pruning mask and pruned state_dict.
    """
    pruned_state_dict = {}
    pruning_mask = {}

    for name, param in model.state_dict().items():
        if not isinstance(param, torch.Tensor) or not param.is_floating_point():
            pruned_state_dict[name] = param
            continue

        param_flat = param.view(-1)
        num_weights = param_flat.numel()
        num_pruned = int(pruning_ratio * num_weights)

        if least_relevant_first:
            sorted_indices = torch.argsort(torch.abs(param_flat))
            prune_indices = sorted_indices[:num_pruned]
        else:
            prune_indices = torch.randperm(num_weights)[:num_pruned]

        mask = torch.ones_like(param_flat)
        mask[prune_indices] = 0
        mask = mask.view_as(param)

        pruning_mask[name] = mask
        pruned_state_dict[name] = param * mask

        if check_nan:
            if torch.isnan(pruned_state_dict[name]).any() or torch.isinf(pruned_state_dict[name]).any():
                print(f"NaN or Inf detected in pruned parameter: {name}")
                raise ValueError(f"Pruned parameter {name} contains NaN or Inf values.")

        if verbose:
            num_pruned_actual = torch.sum(mask == 0).item()
            print(
                f"Layer: {name} | Mask shape: {mask.shape} | "
                f"Pruned: {num_pruned_actual}/{num_weights} ({pruning_ratio:.2f})"
            )

    return {"pruned_state_dict": pruned_state_dict, "pruning_mask": pruning_mask}




def apply_lrp_pruning(model, composite, component_attributor, pruning_rate=0.3, train_loader=None, device=None):
    """
    Apply LRP-based pruning to the model using relevance scores.

    Args:
        model (torch.nn.Module): The model to prune.
        composite: The composite rules for LRP.
        component_attributor: The attribution handler.
        pruning_rate (float): The fraction of parameters to prune.
        train_loader (torch.utils.data.DataLoader): The DataLoader used for relevance calculation.
        device (torch.device): The device to perform computations on.
    """
    if train_loader is None or device is None:
        raise ValueError("Both 'train_loader' and 'device' must be provided when applying LRP pruning.")

    # Calculate relevances using LRP
    print("Computing LRP relevances...")
    components_relevances = component_attributor.attribute(
        model=model,
        dataloader=(
            (batch[1].float().to(device), batch[4].to(device))  # Explicitly move data and labels to device
            for batch in train_loader
        ),
        composite=composite,
        abs_flag=True,
        device=device
    )

    print(f"Relevance scores calculated for components.")

    # Generate pruning mask using the calculated relevances
    pruning_mask = GlobalPruningOperations.generate_global_pruning_mask(
        model=model,
        global_concept_maps=components_relevances,
        pruning_percentage=pruning_rate,
        least_relevant_first=True,
        device=device
    )

    print(f"Generated LRP-based pruning mask.")

    # Apply the generated pruning mask to the model
    state_dict = model.state_dict()
    for name, mask in pruning_mask.items():
        if name in state_dict:
            print(f"Applying pruning mask to layer: {name}")
            state_dict[name] = state_dict[name] * mask["weight"]
        else:
            print(f"Skipping layer {name}, not found in model.")
    model.load_state_dict(state_dict)

    print("LRP pruning applied successfully.")
    return pruning_mask

