import torch
import numpy as np

def one_hot(tensor, targets):
    """
    Generate a one-hot encoded tensor.

    Args:
        tensor (torch.Tensor): The model output tensor.
        targets (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    one_hot_tensor = torch.zeros_like(tensor)
    one_hot_tensor.scatter_(1, targets.view(-1, 1), 1.0)
    return one_hot_tensor

def one_hot_max(tensor, targets):
    """
    Generate a one-hot encoded tensor for the maximum activations.

    Args:
        tensor (torch.Tensor): The model output tensor.
        targets (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: One-hot encoded tensor with the highest activation set to 1.
    """
    one_hot_tensor = torch.zeros_like(tensor)
    max_indices = tensor.argmax(dim=1, keepdim=True)
    one_hot_tensor.scatter_(1, max_indices, 1.0)
    return one_hot_tensor

class ModelLayerUtils:
    @staticmethod
    def get_module_from_name(model, name):
        """
        Retrieve a module from a model by its name.

        Args:
            model (torch.nn.Module): The model.
            name (str): The name of the layer.

        Returns:
            torch.nn.Module: The requested module.
        """
        modules = name.split(".")
        submodule = model
        for module in modules:
            submodule = getattr(submodule, module)
        return submodule

    @staticmethod
    def get_layer_names(model, target_layer_types):
        """
        Get all layer names of a specific type from a model.

        Args:
            model (torch.nn.Module): The model.
            target_layer_types (list): List of layer types to search for.

        Returns:
            list: Names of all matching layers.
        """
        layer_names = []
        for name, module in model.named_modules():
            if type(module) in target_layer_types:
                layer_names.append(name)
        return layer_names
