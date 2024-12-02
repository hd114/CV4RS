from collections import OrderedDict

import torch
import torch.nn.utils.prune as prune

from pxp.utils import ModelLayerUtils


class LocalPruningOperations:
    def __init__(self):
        pass

    def bind_mask_to_module(
            self,
            model,
            layer_name,
            pruning_mask,
            weight_or_bias,
            remove_re_parametrization=True,
    ):
        """
        Bind pruning mask to the module using PyTorch's pruning APIs.
        """
        module = ModelLayerUtils.get_module_from_name(model, layer_name)
        if module is None:
            raise ValueError(f"Layer {layer_name} not found in the model.")

        # Debugging information about the layer and mask
        print(f"[DEBUG] Validating mask before application for layer: {layer_name}")
        print(f"  Layer weight shape: {module.weight.shape if hasattr(module, 'weight') else 'No weight attribute'}")
        print(f"  Layer bias shape: {module.bias.shape if hasattr(module, 'bias') else 'No bias attribute'}")
        print(
            f"  Mask weight shape: {pruning_mask.shape if isinstance(pruning_mask, torch.Tensor) else 'Mask is not a Tensor'}")

        # Ensure the mask is reshaped to match the weight if needed
        if isinstance(pruning_mask, torch.Tensor) and hasattr(module, 'weight'):
            if pruning_mask.numel() == module.weight.numel():
                pruning_mask = pruning_mask.view_as(module.weight)
                print(f"[INFO] Mask reshaped to match layer weight dimensions.")
            else:
                print(
                    f"[ERROR] Mask dimensions {pruning_mask.shape} do not match weight dimensions {module.weight.shape}.")
                raise ValueError(f"Cannot apply mask to layer {layer_name}.")

        # Debugging adjusted mask shape
        print(f"[DEBUG] Adjusted mask shape: {pruning_mask.shape}")

        # Apply the mask using PyTorch's pruning API
        try:
            prune.custom_from_mask(
                module,
                weight_or_bias,
                mask=pruning_mask,
            )
            print(f"[INFO] Successfully applied mask to layer: {layer_name}")
        except Exception as e:
            print(f"[ERROR] Pruning application failed for layer {layer_name}: {e}")
            raise

        # Optionally remove the re-parametrization
        if remove_re_parametrization:
            prune.remove(module, weight_or_bias)

    def fit_pruning_mask(self, model, layer_name, pruning_mask):
        """
        Apply the pruning mask to the model and fix it.

        Args:
            model (torch.nn.Module): The model to prune.
            layer_name (str): The layer to which the pruning mask is applied.
            pruning_mask (dict): Dictionary containing the binary mask for the concepts to prune.
        """
        layer = ModelLayerUtils.get_module_from_name(model, layer_name)
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found in the model.")

        mask_keys = list(pruning_mask.keys())
        for layer_type in mask_keys:
            if layer_type not in ["Conv2d", "BatchNorm2d"]:
                continue

            # Debugging output before mask adjustment
            '''print(f"[DEBUG] Processing layer: {layer_name}, Type: {layer_type}")
            print(f"[DEBUG] Original mask shape: {pruning_mask[layer_type]['weight'].shape}")
            print(f"[DEBUG] Layer weight shape: {layer.weight.shape}")'''

            # Retrieve weight mask
            weight_mask = pruning_mask[layer_type]["weight"]
            if isinstance(weight_mask, torch.Tensor):
                # Case 1: Mask matches filter count (1D)
                if weight_mask.numel() == layer.weight.shape[0]:
                    print(f"[INFO] Reshaping 1D mask to match 4D weight dimensions.")
                    weight_mask = weight_mask.view(-1, 1, 1, 1).expand_as(layer.weight)

                # Case 2: Mask matches total element count (4D)
                elif weight_mask.numel() == layer.weight.numel():
                    print(f"[INFO] Reshaping flat mask to match exact weight dimensions.")
                    weight_mask = weight_mask.view_as(layer.weight)

                # Error case: Mask dimensions are invalid
                else:
                    raise ValueError(
                        f"[ERROR] Cannot reshape mask: {weight_mask.shape} -> {layer.weight.shape}. "
                        f"Check the pruning mask generation logic."
                    )

                # Update the pruning mask
                pruning_mask[layer_type]["weight"] = weight_mask
                print(f"[DEBUG] Final adjusted mask shape: {weight_mask.shape}")

            # Bind the adjusted mask
            self.bind_mask_to_module(
                model,
                layer_name,
                pruning_mask[layer_type]["weight"],
                weight_or_bias="weight",
                remove_re_parametrization=True,
            )

            # Process bias if it exists
            if layer.bias is not None and "bias" in pruning_mask[layer_type]:
                print(f"[DEBUG] Layer {layer_name} has bias with shape {layer.bias.shape}")

                bias_mask = pruning_mask[layer_type]["bias"]
                if isinstance(bias_mask, torch.Tensor):
                    if bias_mask.numel() == layer.bias.numel():
                        print(f"[INFO] Adjusting bias mask to match bias dimensions.")
                        bias_mask = bias_mask.view_as(layer.bias)
                        pruning_mask[layer_type]["bias"] = bias_mask
                    else:
                        raise ValueError(
                            f"[ERROR] Bias mask shape {bias_mask.shape} does not match layer bias shape {layer.bias.shape}."
                        )

                self.bind_mask_to_module(
                    model,
                    layer_name,
                    pruning_mask[layer_type]["bias"],
                    weight_or_bias="bias",
                    remove_re_parametrization=True,
                )

    def generate_local_pruning_mask(
            self,
            pruning_mask_shape,
            pruning_indices,
            subsequent_layer_pruning,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Generate a binary pruning mask for the specified layer types.

        Args:
            pruning_mask_shape (tuple): Shape of the pruning mask.
            pruning_indices (list): Indices of the concepts to prune.
            subsequent_layer_pruning (str): Specifies which layers to prune.
                Options: ["Conv2d", "Both", "BatchNorm2d", "Linear", "Softmax"].
            device (str): Device to run on ("cuda" or "cpu").

        Returns:
            dict: Binary pruning mask for the specified layer types.
        """
        # Initialize masks
        final_pruning_mask = {}

        if subsequent_layer_pruning == "Linear":
            linear_weight_mask = torch.ones(pruning_mask_shape).to(device)
            linear_bias_mask = torch.ones(pruning_mask_shape[0]).to(device)
            # Apply pruning
            linear_weight_mask[pruning_indices] = 0
            linear_bias_mask[pruning_indices] = 0
            final_pruning_mask["Linear"] = {
                "weight": linear_weight_mask,
                "bias": linear_bias_mask,
            }

        if subsequent_layer_pruning in ["Conv2d", "Both"]:
            cnn_weight_mask = torch.ones(pruning_mask_shape).to(device)
            cnn_bias_mask = torch.ones(pruning_mask_shape[0]).to(device)
            # Apply pruning
            cnn_weight_mask[pruning_indices] = 0
            cnn_bias_mask[pruning_indices] = 0
            final_pruning_mask["Conv2d"] = {
                "weight": cnn_weight_mask,
                "bias": cnn_bias_mask,
            }

        if subsequent_layer_pruning in ["BatchNorm2d", "Both"]:
            bn_weight_mask = torch.ones(pruning_mask_shape[0]).to(device)
            bn_bias_mask = torch.ones(pruning_mask_shape[0]).to(device)
            # Apply pruning
            bn_weight_mask[pruning_indices] = 0
            bn_bias_mask[pruning_indices] = 0
            final_pruning_mask["BatchNorm2d"] = {
                "weight": bn_weight_mask,
                "bias": bn_bias_mask,
            }

        if subsequent_layer_pruning == "Softmax":
            softmax_weight_mask = torch.ones(pruning_mask_shape).to(device)
            # Apply pruning
            softmax_weight_mask[pruning_indices] = 0
            final_pruning_mask["Softmax"] = {"weight": softmax_weight_mask}

        # Return the constructed mask dictionary
        if not final_pruning_mask:
            raise ValueError(
                f"Invalid option for subsequent_layer_pruning: {subsequent_layer_pruning}"
            )
        return final_pruning_mask


class GlobalPruningOperations(LocalPruningOperations):
    def __init__(self, target_layer, layer_names, device=None):
        self.target_layer = target_layer
        self.layer_names = layer_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

    # def generate_global_pruning_mask(
    #     self,
    #     model,
    #     global_concept_maps,
    #     pruning_precentage,
    #     subsequent_layer_pruning="Conv2d",
    #     least_relevant_first=True,
    #     device="cuda",
    # ):
    #     """
    #     Generate a global pruning mask for the model based on the LRP relevances
    #
    #     Args:
    #         mode (str): whether to use "Relevance", "Activation", or "Random" for concept attribution
    #         model (torch.module): the model
    #         pruning_precentage (float): the precentage of the concepts to prune
    #         dataset (torchvision.datasets): the dataset which holds tbe images used for pruning
    #         sample_indices (list): indices of the samples to use for pruning from the dataset
    #         device (str, optional): device to run on. Defaults to "cuda".
    #         subsequent_layer_pruning (str, optional): Whether or not to prune the subsequent layers(BatchNorm). Options are ["Conv2d", "Both", "BatchNorm2d"].
    #                                                         When "Conv2d" is chosen, only the Conv2d will be pruned. When "Both" is chosen, both the batchnorm
    #                                                         and the subsequent BatchNorm will be pruned. But if "BatchNorm2d" is chosen, only
    #                                                         the BatchNorm of the subsequent layer will be pruned, not the Conv2d! Defaults to None.
    #         lrp_composite (zennit.composites, optional): LRP composites used for Relevance-Based Concept Pruning. Defaults to EpsilonPlusFlat.
    #         abs_norm (bool, optional): Whether or not to use absolute normalization when computing the concept relevences via zennit-crp. Defaults to True.
    #         abs_sort (bool, optional): Whether or not to sort the concepts by their absolute value. Defaults to True.
    #         least_relevant_first (bool, optional): Whether or not to prune the least values(Relevance/Activation)
    #                                               or the most values(Relevance/Activation). Defaults to True(least).
    #
    #     Returns:
    #         dict: Dictionary of binary mask of the concepts to prune for each layer
    #     """
    #     global_pruning_masks_shapes = OrderedDict([])
    #     if self.target_layer != torch.nn.Softmax:
    #         for layer_name in self.layer_names:
    #             global_pruning_masks_shapes[layer_name] = (
    #                 ModelLayerUtils.get_module_from_name(model, layer_name).weight.shape
    #             )
    #
    #     interval_indices = OrderedDict([])
    #     old_start_index = 0
    #     for layer_name in self.layer_names:
    #         if layer_name not in interval_indices.keys():
    #             interval_indices[layer_name] = (
    #                 old_start_index,
    #                 old_start_index + global_concept_maps[layer_name].shape[0] - 1,
    #             )
    #             old_start_index += global_concept_maps[layer_name].shape[0]
    #
    #     # Generate the indices of concepts/filters to prune from each layer
    #     global_pruning_indices = self.generate_global_pruning_indices(
    #         global_concept_maps,
    #         interval_indices,
    #         pruning_precentage,
    #         least_relevant_first,
    #     )
    #
    #     # Generate the pruning masks for each layer
    #     if self.target_layer != torch.nn.Softmax:
    #         global_pruning_mask = OrderedDict([])
    #         for layer_name, layer_pruning_indices in global_pruning_indices.items():
    #
    #             global_pruning_mask[layer_name] = self.generate_local_pruning_mask(
    #                 global_pruning_masks_shapes[layer_name],
    #                 layer_pruning_indices,
    #                 subsequent_layer_pruning=subsequent_layer_pruning,
    #                 device=device,
    #             )
    #
    #         return global_pruning_mask
    #
    #     else:
    #         return global_pruning_indices

    def generate_global_pruning_mask(
            self,
            model,
            global_concept_maps,
            pruning_percentage,
            least_relevant_first=True,
            device=None
    ):
        """
        Generate a global pruning mask based on relevance maps.

        Args:
            model (torch.nn.Module): The model to prune.
            global_concept_maps (dict): Relevance maps for each layer.
            pruning_percentage (float): Fraction of parameters to prune.
            least_relevant_first (bool): Whether to prune the least relevant parameters.
            device (str): Device to allocate masks.

        Returns:
            dict: Pruning masks for all relevant layers.
        """
        # Ensure global_concept_maps is in the correct format
        assert isinstance(global_concept_maps, dict), "global_concept_maps must be a dictionary."

        global_pruning_masks = OrderedDict()
        for layer_name, relevance_map in global_concept_maps.items():
            print(f"Layer: {layer_name}")
            print(f"Type of relevance_map: {type(relevance_map)}")
            print(f"Content of relevance_map: {relevance_map}")

            # Falls relevance_map ein dict ist, die relevanten Daten extrahieren
            if isinstance(relevance_map, dict):
                if "Conv2d" in relevance_map:
                    relevance_map = relevance_map["Conv2d"]["weight"]
                elif "BatchNorm2d" in relevance_map:
                    relevance_map = relevance_map["BatchNorm2d"]["weight"]
                else:
                    raise ValueError(f"Unsupported structure in relevance_map for layer {layer_name}: {relevance_map}")

            # Ensure the relevance map is a tensor
            if not isinstance(relevance_map, torch.Tensor):
                raise TypeError(
                    f"Relevance map for layer {layer_name} must be a tensor after extraction. Found type: {type(relevance_map)}"
                )

            # Compute the number of elements to prune
            num_prune = int(pruning_percentage * relevance_map.numel())

            # Flatten the relevance map and compute the indices to prune
            flat_relevance = relevance_map.view(-1)
            _, prune_indices = flat_relevance.topk(num_prune, largest=not least_relevant_first)

            # Generate local pruning mask
            pruning_mask = self.generate_local_pruning_mask(
                pruning_mask_shape=relevance_map.shape,
                pruning_indices=prune_indices,
                subsequent_layer_pruning="Both",  # or another mode based on your logic
                device=device,
            )
            global_pruning_masks[layer_name] = pruning_mask

        return global_pruning_masks

    def generate_global_pruning_indices(
        self,
        global_concept_maps,
        interval_indices,
        pruning_percentage,
        least_relevant_first,
    ):
        """
        Generate the indices of concepts/filters to prune from each layer

        Args:
            global_concept_maps (dict): summed of concept relevances for each layer
            interval_indices (dict): interval indices of the concepts/filters for each layer
            pruning_percentage (float): percentage of concepts/filters to prune
            least_relevant_first (bool): whether to prune the least or most relevant concepts/filters

        Returns:
            dict: Dictionary of indices of concepts/filters to prune for each layer
        """
        # Flatten relevances for each layer into a single tensor
        flattened_concept_relevances = torch.cat(
            [value.flatten() for value in global_concept_maps.values()]
        )

        # Total number of concepts/filters to prune
        total_num_candidates = int(
            flattened_concept_relevances.shape[0] * pruning_percentage
        )

        # Sort the concepts/filters by their relevances and get the indices
        _, pruning_indices = flattened_concept_relevances.topk(
            total_num_candidates,
            largest=not least_relevant_first,
        )

        # Assign the sorted indices to the corresponding layers
        # ,stating the filters/concepts to prune from each layer
        global_pruning_indices = OrderedDict([])
        for layer_name, _ in global_concept_maps.items():
            start_index, end_index = interval_indices[layer_name]
            global_pruning_indices[layer_name] = (
                pruning_indices[
                    (pruning_indices >= start_index) & (pruning_indices <= end_index)
                ]
                - start_index
            )

        return global_pruning_indices

    def bind_mask_to_module(
            self,
            model,
            layer_name,
            pruning_mask,
            weight_or_bias,
            remove_re_parametrization=True,
    ):
        """
        Bind pruning mask to the module using PyTorch's pruning APIs.

        Args:
            model (torch.nn.Module): The model containing the layer.
            layer_name (str): The name of the layer.
            pruning_mask (torch.Tensor): The mask to apply.
            weight_or_bias (str): Specify whether to prune "weight" or "bias".
            remove_re_parametrization (bool): Whether to remove re-parametrization.
        """
        module = ModelLayerUtils.get_module_from_name(model, layer_name)
        if module is None:
            raise ValueError(f"Layer {layer_name} not found in the model.")

        ''''# Debugging information
        print(f"[DEBUG] Binding mask to module: {layer_name}, Target: {weight_or_bias}")
        print(f"  Module weight shape: {module.weight.shape if hasattr(module, 'weight') else 'No weight'}")
        print(f"  Module bias shape: {module.bias.shape if hasattr(module, 'bias') else 'No bias'}")
        print(f"  Mask shape: {pruning_mask.shape}")'''

        # Get the weight or bias shape
        param_shape = getattr(module, weight_or_bias).shape
        if pruning_mask.shape != param_shape:
            print(f"[WARNING] Adjusting mask shape for {weight_or_bias} in layer: {layer_name}")
            try:
                pruning_mask = pruning_mask.view_as(getattr(module, weight_or_bias))
            except Exception as e:
                raise ValueError(
                    f"Cannot reshape mask for {weight_or_bias} in layer {layer_name}: {e}"
                )

        #print(f"[INFO] Mask reshaped to match {weight_or_bias} dimensions: {pruning_mask.shape}")

        # Apply the mask using PyTorch's pruning API
        try:
            prune.custom_from_mask(module, weight_or_bias, mask=pruning_mask)
            print(f"[INFO] Successfully applied mask to {weight_or_bias} in layer: {layer_name}")
        except Exception as e:
            raise ValueError(f"Pruning application failed for {weight_or_bias} in layer {layer_name}: {e}")

        if remove_re_parametrization:
            prune.remove(module, weight_or_bias)
            #print(f"[INFO] Removed re-parametrization for {weight_or_bias} in layer: {layer_name}")

    @staticmethod
    def mask_attention_head(model, layer_names, head_indices):
        hook_handles = []
        act_layer_softmax = {}

        def generate_set_forward_hook_softmax(layer_name, head_indices):
            def set_out_activations(module, input, output):
                output[:, head_indices] = 0

            return set_out_activations

        def generate_get_forward_hook_softmax(layer_name):
            def get_out_activations(module, input, output):
                act_layer_softmax[layer_name] = output

            return get_out_activations

        # append hook to last activation of mlp layer
        for name, layer in model.named_modules():
            if name == layer_names:
                hook_handles.append(
                    layer.register_forward_hook(
                        generate_set_forward_hook_softmax(name, head_indices)
                    )
                )

        return hook_handles
