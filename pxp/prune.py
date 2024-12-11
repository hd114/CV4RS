from collections import OrderedDict

import torch
import torch.nn.utils.prune as prune

from pxp.utils import ModelLayerUtils

5
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
        prune.custom_from_mask(
            module,
            weight_or_bias,
            mask=pruning_mask,
        )
        if remove_re_parametrization:
            prune.remove(module, weight_or_bias)

    def fit_pruning_mask(self, model, layer_name, pruning_mask):
        """
        Apply the pruning mask to the model and fix it

        Args:
            model (torch.nn.module): the model to prune
            layer_name (str): the layer which the pruning mask is applied to
            pruning_mask (dict): dictionary of binary mask of the concepts to prune for each layer
        """
        mask_keys = list(pruning_mask.keys())
        batch_norm_order_flag = False
        if "BatchNorm2d" in mask_keys:
            if ModelLayerUtils.is_batchnorm2d_after_conv2d(model):
                batch_norm_order_flag = True
                conv_bn_layers = ModelLayerUtils.get_layer_names(
                    model, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
                )
                bn_layer_name = conv_bn_layers[conv_bn_layers.index(layer_name) + 1]

        for layer_type in mask_keys:
            if layer_type == "BatchNorm2d" and batch_norm_order_flag == True:
                layer_name_to_prune = bn_layer_name
            else:
                layer_name_to_prune = layer_name

            # Prune the weights first
            self.bind_mask_to_module(
                model,
                layer_name_to_prune,
                pruning_mask[layer_type]["weight"],
                weight_or_bias="weight",
                remove_re_parametrization=True,
            )

            # For bias, prune if they exist
            if ModelLayerUtils.get_module_from_name(model, layer_name).bias is not None:
                self.bind_mask_to_module(
                    model,
                    layer_name_to_prune,
                    pruning_mask[layer_type]["bias"],
                    weight_or_bias="bias",
                    remove_re_parametrization=True,
                )

    def generate_local_pruning_mask(
        self,
        pruning_mask_shape,
        pruning_indices,
        subsequent_layer_pruning,
        device="cuda",
    ):
        """
        Given the shape of the cumulative pruning mask, generate a binary mask of the concepts to prune further

        Args:
            pruning_mask_shape (tuple): shape of the cumulative pruning mask
            pruning_indices (list): indices of the concepts to prune further
            subsequent_layer_pruning (str, optional): Whether or not to prune the subsequent layers(BatchNorm). Options are ["Conv2d", "Both", "BatchNorm2d"].
            device (str, optional): device to run on. Defaults to "cuda".

        Returns:
            dict: pruning binary mask
        """
        if subsequent_layer_pruning == "Linear":
            final_pruning_mask_linear_weight = torch.ones(pruning_mask_shape).to(device)
            final_pruning_mask_linear_bias = torch.ones(pruning_mask_shape[0]).to(
                device
            )

            final_pruning_mask_linear_weight[pruning_indices] = 0
            final_pruning_mask_linear_bias[pruning_indices] = 0
            return {
                "Linear": {
                    "weight": final_pruning_mask_linear_weight,
                    "bias": final_pruning_mask_linear_bias,
                }
            }
        if subsequent_layer_pruning in ["Conv2d", "Both"]:
            final_pruning_mask_cnn_weight = torch.ones(pruning_mask_shape).to(device)
            final_pruning_mask_cnn_bias = torch.ones(pruning_mask_shape[0]).to(device)
            # Zero out the elected concepts
            final_pruning_mask_cnn_weight[pruning_indices] = 0
            final_pruning_mask_cnn_bias[pruning_indices] = 0
        if subsequent_layer_pruning in ["Both", "BatchNorm2d"]:
            final_pruning_mask_bn_weight = torch.ones(pruning_mask_shape[0]).to(device)
            final_pruning_mask_bn_bias = torch.ones(pruning_mask_shape[0]).to(device)
            # Zero out the elected concepts
            final_pruning_mask_bn_weight[pruning_indices] = 0
            final_pruning_mask_bn_bias[pruning_indices] = 0
        if subsequent_layer_pruning in ["Softmax"]:
            final_pruning_mask_softmax_weight = torch.ones(pruning_mask_shape).to(
                device
            )
            # Zero out the elected concepts
            final_pruning_mask_softmax_weight[pruning_indices] = 0

        if subsequent_layer_pruning == "Conv2d":
            return {
                "Conv2d": {
                    "weight": final_pruning_mask_cnn_weight,
                    "bias": final_pruning_mask_cnn_bias,
                }
            }
        elif subsequent_layer_pruning == "Both":
            return {
                "Conv2d": {
                    "weight": final_pruning_mask_cnn_weight,
                    "bias": final_pruning_mask_cnn_bias,
                },
                "BatchNorm2d": {
                    "weight": final_pruning_mask_bn_weight,
                    "bias": final_pruning_mask_bn_bias,
                },
            }
        elif subsequent_layer_pruning == "BatchNorm2d":
            return {
                "BatchNorm2d": {
                    "weight": final_pruning_mask_bn_weight,
                    "bias": final_pruning_mask_bn_bias,
                }
            }
        elif subsequent_layer_pruning == "Softmax":
            return {"Softmax": {"weight": final_pruning_mask_softmax_weight}}


class GlobalPruningOperations(LocalPruningOperations):
    def __init__(self, target_layer, layer_names):
        self.target_layer = target_layer
        self.layer_names = layer_names

    def generate_global_pruning_mask(
        self,
        model,
        global_concept_maps,
        pruning_percentage,
        subsequent_layer_pruning="Conv2d",
        least_relevant_first=True,
        device="cuda",
    ):
        """
        Generate a global pruning mask for the model based on the LRP relevances

        Args:
            mode (str): whether to use "Relevance", "Activation", or "Random" for concept attribution
            model (torch.module): the model
            pruning_percentage (float): the precentage of the concepts to prune
            dataset (torchvision.datasets): the dataset which holds tbe images used for pruning
            sample_indices (list): indices of the samples to use for pruning from the dataset
            device (str, optional): device to run on. Defaults to "cuda".
            subsequent_layer_pruning (str, optional): Whether or not to prune the subsequent layers(BatchNorm). Options are ["Conv2d", "Both", "BatchNorm2d"].
                                                            When "Conv2d" is chosen, only the Conv2d will be pruned. When "Both" is chosen, both the batchnorm
                                                            and the subsequent BatchNorm will be pruned. But if "BatchNorm2d" is chosen, only
                                                            the BatchNorm of the subsequent layer will be pruned, not the Conv2d! Defaults to None.
            lrp_composite (zennit.composites, optional): LRP composites used for Relevance-Based Concept Pruning. Defaults to EpsilonPlusFlat.
            abs_norm (bool, optional): Whether or not to use absolute normalization when computing the concept relevances via zennit-crp. Defaults to True.
            abs_sort (bool, optional): Whether or not to sort the concepts by their absolute value. Defaults to True.
            least_relevant_first (bool, optional): Whether or not to prune the least values(Relevance/Activation)
                                                  or the most values(Relevance/Activation). Defaults to True(least).

        Returns:
            dict: Dictionary of binary mask of the concepts to prune for each layer
        """
        global_pruning_masks_shapes = OrderedDict([])
        if self.target_layer != torch.nn.Softmax:
            for layer_name in self.layer_names:
                global_pruning_masks_shapes[layer_name] = (
                    ModelLayerUtils.get_module_from_name(model, layer_name).weight.shape
                )

        interval_indices = OrderedDict([])
        old_start_index = 0
        for layer_name in self.layer_names:
            if layer_name not in interval_indices.keys():
                interval_indices[layer_name] = (
                    old_start_index,
                    old_start_index + global_concept_maps[layer_name].shape[0] - 1,
                )
                old_start_index += global_concept_maps[layer_name].shape[0]

        # Generate the indices of concepts/filters to prune from each layer
        global_pruning_indices = self.generate_global_pruning_indices(
            global_concept_maps,
            interval_indices,
            pruning_percentage,
            least_relevant_first,
        )

        # Generate the pruning masks for each layer
        if self.target_layer != torch.nn.Softmax:
            global_pruning_mask = OrderedDict([])
            for layer_name, layer_pruning_indices in global_pruning_indices.items():

                global_pruning_mask[layer_name] = self.generate_local_pruning_mask(
                    global_pruning_masks_shapes[layer_name],
                    layer_pruning_indices,
                    subsequent_layer_pruning=subsequent_layer_pruning,
                    device=device,
                )

            return global_pruning_mask

        else:
            return global_pruning_indices

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

    def fit_pruning_mask(self, model, layer_name, pruning_mask):
        """
        Apply the pruning mask to the model and fix it

        Args:
            model (torch.nn.module): the model to prune
            layer_name (str): the layer which the pruning mask is applied to
            pruning_mask (dict): dictionary of binary mask of the concepts to prune for each layer
        """
        mask_keys = list(pruning_mask.keys())
        batch_norm_order_flag = False
        if "BatchNorm2d" in mask_keys:
            if ModelLayerUtils.is_batchnorm2d_after_conv2d(model):
                batch_norm_order_flag = True
                conv_bn_layers = ModelLayerUtils.get_layer_names(
                    model, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
                )
                bn_layer_name = conv_bn_layers[conv_bn_layers.index(layer_name) + 1]

        for layer_type in mask_keys:
            if layer_type == "BatchNorm2d" and batch_norm_order_flag == True:
                layer_name_to_prune = bn_layer_name
            else:
                layer_name_to_prune = layer_name

            # Prune the weights first
            self.bind_mask_to_module(
                model,
                layer_name_to_prune,
                pruning_mask[layer_type]["weight"],
                weight_or_bias="weight",
                remove_re_parametrization=True,
            )

            # For bias, prune if they exist
            if ModelLayerUtils.get_module_from_name(model, layer_name).bias is not None:
                self.bind_mask_to_module(
                    model,
                    layer_name_to_prune,
                    pruning_mask[layer_type]["bias"],
                    weight_or_bias="bias",
                    remove_re_parametrization=True,
                )


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
