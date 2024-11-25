from collections import OrderedDict
import torch
from pxp.utils import one_hot_max, one_hot, ModelLayerUtils

class LatentRelevanceAttributor:
    def __init__(self, layers_list_to_track) -> None:
        """
        Constructor

        Args:
            layers_list_to_track (list): list of types of layers to track
        """
        self.layers_list_to_track = layers_list_to_track
        self.latent_output = {}

    def lrp_pass(self, model, inputs, targets, composite, initial_relevance, device):
        """
        Compute the relevance using LRP

        Args:
            model (torch.nn.module): the model to be explained
            inputs (torch.tensor): inputs or the given images
            targets (torch.tensor): targets of the given images
            composite (): LRP composite
            device (): device to be used

        Returns:
            (torch.tensor): the computed heatmap using LRP
        """
        if initial_relevance == 1:
            initial_relevance_function = one_hot
        elif initial_relevance == "logit":
            initial_relevance_function = one_hot_max

        with torch.enable_grad():
            inputs.requires_grad = True
            if composite:
                with composite.context(model) as modified_model:
                    relevance = self.compute_relevance(
                        modified_model,
                        inputs,
                        targets,
                        initial_relevance_function,
                        device,
                    )
            else:
                relevance = self.compute_relevance(
                    model, inputs, targets, initial_relevance_function, device
                )

        self.remove_hooks()
        self.parse_latent_relevances(model)

        return relevance

    def compute_relevance(
        self, model, inputs, targets, initial_relevance_function, device
    ):
        self.clear_latent_info()
        self.hook_handles = self.register_hooks(model)

        print(f"Input shape before model forward: {inputs.shape}")
        if len(inputs.shape) < 4:
            print(f"Unexpected input shape for LRP: {inputs.shape}. Adding dummy dimensions.")
            inputs = inputs.unsqueeze(0).unsqueeze(0)
            print(f"Modified input shape for LRP: {inputs.shape}")

        # Adjust input channels if necessary
        first_layer_channels = model.encoder[0].weight.shape[1]
        if inputs.shape[1] != first_layer_channels:
            print(f"Adjusting input channels from {inputs.shape[1]} to {first_layer_channels}")
            inputs = inputs.repeat(1, first_layer_channels, 1, 1)

        # Adjust batch size to match targets
        if inputs.shape[0] != targets.shape[0]:
            print(f"Adjusting input batch size from {inputs.shape[0]} to match targets batch size {targets.shape[0]}")
            inputs = inputs.repeat(targets.shape[0], 1, 1, 1)

        print(f"Input shape before relevance computation: {inputs.shape}")

        try:
            output = model(inputs)
            print(f"Model output shape: {output.shape}")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            raise

        if targets.dtype != torch.long:
            print(f"Converting targets from {targets.dtype} to torch.long")
            targets = targets.long()

        if targets.shape[0] != output.shape[0]:
            print(
                f"Mismatch between targets batch size ({targets.shape[0]}) and model output batch size ({output.shape[0]})."
            )
            raise ValueError("Batch size mismatch between targets and model output.")

        if len(targets.shape) > 1 and targets.shape[1] == output.shape[-1]:
            print(f"Reducing targets shape from {targets.shape} to {targets.shape[0]} for one-hot encoding.")
            targets = torch.argmax(targets, dim=1)

        try:
            grad_outputs = initial_relevance_function(output, targets).to(device)

            print(f"Grad output shape: {grad_outputs.shape}")
            if grad_outputs.shape != output.shape:
                raise ValueError(
                    f"Mismatch in shape: grad_outputs {grad_outputs.shape} and model output {output.shape}"
                )

            (relevance,) = torch.autograd.grad(
                outputs=output,
                inputs=inputs,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
            )
            print(f"Relevance computation successful. Relevance shape: {relevance.shape}")
        except Exception as e:
            print(f"Error during relevance computation: {e}")
            raise

        relevance = relevance.detach().cpu()

        if torch.isnan(relevance).any() or torch.isinf(relevance).any():
            print(f"NaN or Inf detected in relevance values.")
            raise ValueError("Relevance contains invalid values (NaN or Inf).")

        return relevance

    def remove_hooks(self):
        """
        Remove the hooks
        """
        for handle in self.hook_handles:
            handle.remove()

    def register_hooks(self, model):
        """
        Attach hooks to the modules

        Args:
            model (torch.nn.module): model

        Returns:
            list: handles of the forward hooks
        """
        hook_handles = []
        for layer_name in self.layers_list_to_track:
            for name, module in model.named_modules():
                if name == layer_name:
                    hook_handles.append(
                        module.register_forward_hook(
                            self.get_hook_function(name, self.latent_output)
                        )
                    )

        return hook_handles

    @staticmethod
    def get_hook_function(layer_name, layer_out):
        """
        Static method to get the hook function

        Args:
            layer_name (str): layer_name
            layer_out (dict): dictionary to store the output of the layers
        """

        def forward_hook_function(module, input, output):
            layer_out[layer_name] = output
            output.retain_grad()

        return forward_hook_function

    def parse_latent_relevances(self, model):
        """
        Extract the relevance values from the output tensors
        via .grad

        Args:
            model (torch.nn.module): model
        """
        self.latent_relevances = {}
        for layer_name in self.layers_list_to_track:
            for name, module in model.named_modules():
                if name == layer_name:
                    self.latent_relevances[name] = self.latent_output[name].grad.detach().cpu()

    def clear_latent_info(self):
        self.latent_output = {}
        self.latent_relevances = {}
