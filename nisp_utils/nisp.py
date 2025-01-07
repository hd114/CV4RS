import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def nisp_conv2d(conv_module: nn.Conv2d, S_out: torch.Tensor) -> torch.Tensor:
    """
    Propagate output layer importance scores back to input layer importance scores for a given Conv2d module,
    handling arbitrary stride, padding, and dilation. Based on transposed convolution.

    Args:
        conv_module (nn.Conv2d):
            The convolution module over which importance scores are to be propagated back from output layer to input layer.
        S_out (torch.Tensor):
            The output layer importance scores of shape (C_out, H_out, W_out).

    Returns:
        S_in (torch.Tensor):
            Input layer importance scores of shape (C_in, H_in, W_in), where H_in and W_in are inferred from
            conv parameters and S_out size.
    """
    # Conv2d weights are shaped (C_out, C_in, kH, kW).
    # For transposed convolution, we need (C_in, C_out, kH, kW).
    W_transpose = torch.abs(conv_module.weight).permute(1, 0, 2, 3).contiguous()               # TAKE ABSOLUTE VALUES!!!

    # Insert batch dimension for S_out so shape is (N=1, C_out, H_out, W_out).
    S_out_4d = S_out.unsqueeze(0)

    # Perform transposed convolution with:
    #   - The "swapped" weight
    #   - No bias
    #   - The same stride/padding/dilation as forward conv
    #   - output_padding=0 (usually fine for matching direct backward pass)
    S_in_4d = F.conv_transpose2d(
        S_out_4d,
        W_transpose,
        bias=None,
        stride=conv_module.stride,
        padding=conv_module.padding,
        output_padding=0,
        dilation=conv_module.dilation,
        groups=1
    )

    # Remove the batch dimension, leaving (C_in, H_in, W_in).
    S_in = S_in_4d.squeeze(0)
    return S_in

def nisp_maxpool2d(pool_module: nn.MaxPool2d, S_out: torch.Tensor) -> torch.Tensor:
    """
    NISP for MaxPool2d using a grouped transposed convolution
    to uniformly distribute output importance across each pooling region.
    (Not only the pooled activations receive scores!! pooled and unpooled treated equally!!!)

    Args:
        pool_module (nn.MaxPool2d):
            The pooling layer from which we extract kernel_size, stride,
            padding, and dilation.
        S_out (torch.Tensor):
            Output importance of shape (C, H_out, W_out).  (No batch dim.)

    Returns:
        S_in (torch.Tensor):
            Input importance of shape (C, H_in, W_in).
    """

    # PHASE 1) Read pooling parameters (kernel_size, stride, padding, dilation)
    #    Each can be int or tuple, so handle both.
    if isinstance(pool_module.kernel_size, int):
        kH, kW = pool_module.kernel_size, pool_module.kernel_size
    else:
        kH, kW = pool_module.kernel_size

    if isinstance(pool_module.stride, int):
        sH, sW = pool_module.stride, pool_module.stride
    else:
        sH, sW = pool_module.stride

    if isinstance(pool_module.padding, int):
        pH, pW = pool_module.padding, pool_module.padding
    else:
        pH, pW = pool_module.padding

    if isinstance(pool_module.dilation, int):
        dH, dW = pool_module.dilation, pool_module.dilation
    else:
        dH, dW = pool_module.dilation

    C, H_out, W_out = S_out.shape

    # PHASE 2) Build transposed-conv kernel of all ones, shaped (C, 1, kH, kW),
    #    and set groups=C so each channel is handled separately.
    weight_ones = torch.ones(                                                                  # torch.abs(ones) == ones
        (C, 1, kH, kW),
        dtype=S_out.dtype,
        device=S_out.device
    )

    # We treat S_out as (N=1, C, H_out, W_out) for the transposed convolution.
    S_out_4d = S_out.unsqueeze(0)

    #PHASE 3) Perform grouped transposed convolution:
    #    - No bias
    #    - groups=C  ensures per-channel “stamping”
    S_in_4d = F.conv_transpose2d(
        S_out_4d,
        weight_ones,
        bias=None,
        stride=(sH, sW),
        padding=(pH, pW),
        output_padding=0,
        groups=C,
        dilation=(dH, dW)
    )
    # S_in_4d now has shape (1, C, H_in, W_in), but each (kH,kW) block is a sum of S_out.
    # We want a uniform distribution, so we divide by (kH*kW).
    S_in_4d /= (kH * kW)                                                                        #Actually can be ignored

    # Remove the batch dimension -> (C, H_in, W_in)
    S_in = S_in_4d.squeeze(0)

    return S_in

def nisp_adaptiveavgpool2d_autograd(
    pool_module: nn.AdaptiveAvgPool2d,
    S_out: torch.Tensor,
    input_shape: tuple
) -> torch.Tensor:
    """
    Propagate importance over AdaptiveAvgPool2d through torch autograd.
    On dummy input with desired shape perform forward pass, then
    call backward with S_out as the gradient. Resulting .grad is the input layer importance.

    Args:
        pool_module (nn.AdaptiveAvgPool2d):
            The adaptive pooling layer whose backward pass we want to mimic.
        S_out (torch.Tensor):
            The output importance, shape (C, H_out, W_out).
            (No batch dimension: one single input.)
        input_shape (tuple):
            The shape of the original input, (C, H_in, W_in).

    Returns:
        S_in (torch.Tensor):
            The propagated importance, shape (C, H_in, W_in).
    """
    C, H_in, W_in = input_shape
    C_out, H_out, W_out = S_out.shape
    assert C == C_out, "Channel mismatch between S_out and input_shape"

    # 1) Create a dummy input with requires_grad=True
    #    We'll add a batch dimension of 1 => (1, C, H_in, W_in).
    dummy_input = torch.zeros(
        (1, C, H_in, W_in),
        dtype=S_out.dtype,
        device=S_out.device,
        requires_grad=True
    )

    # 2) Forward pass through the AdaptiveAvgPool2d
    out = pool_module(dummy_input)  # shape => (1, C, H_out, W_out)

                                                                        # BACKWARD OVER AVGPOOL SHOULD NOT CHANGE SIGN
    # 3) Backward pass: treat S_out as the "gradient" from above
    out.backward(S_out.unsqueeze(0))  # S_out => (C, H_out, W_out), unsqueeze => (1, C, H_out, W_out)

    # 4) The gradient w.r.t. dummy_input is exactly our importance
    S_in = dummy_input.grad.detach().squeeze(0)  # => shape (C, H_in, W_in)

    return S_in

def nisp_overview_old(layer, S_out, input_shape):
    """
    Dispatches to the correct NISP-implementation for module types
    """
    if isinstance(layer, nn.Conv2d):
        # Option A: Use the conv_transpose2d approach for NISP
        return nisp_conv2d(layer, S_out)
        # instead of papers manual loop version

    elif isinstance(layer, nn.MaxPool2d):
        # NISP paper does 'uniform distribution' for max-pool, so:
        return nisp_maxpool2d(layer, S_out)
        # cannot rely on autograd alone, because it would only
        # backprop to the pooled activation

    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        # Option A: use the manual bin-boundaries approach
        # Option B: use the autograd-based approach (better)
        return nisp_adaptiveavgpool2d_autograd(layer, S_out, input_shape)

    else:
        raise NotImplementedError(f"Layer type {type(layer)} not handled.")

def zero_out_smallest_scores(importance_scores, pruning_rate):
    """
    Zeroes out the smallest importance scores at rate pruning_rate.

    Args:
        importance_scores (torch.Tensor): The input tensor.
        pruning_rate (float): The rate at which to zero out small scores (0.0-0.99).

    Returns:
        torch.Tensor: Modified tensor with smallest r% elements zeroed out.
    """
    scores_flattened = importance_scores.flatten()
    k = int(len(scores_flattened) * pruning_rate)  # Number of elements to zero out

    if k > 0:
        threshold = torch.kthvalue(scores_flattened, k).values  # Get the k-th smallest value
        mask = scores_flattened >= threshold  # Mask: Keep only values >= threshold
        scores_flattened = scores_flattened * mask  # Zero out smallest r% values

    return scores_flattened.view_as(importance_scores)


def frl_mag(final_response_layer, pruning_rate = 0 ):
    # TODO prune mag
    return torch.abs(final_response_layer.weight).sum(dim=0)

def inf_fs():
    pass

def nisp(model, S_out, dummy_image_size=(1000,1,1)):
    nisp_blocks = model.list_nisp_blocks() # encoder.named_children() is not sufficient!

    mask = None #TODO create based on provided LRP pruning

    for name, block in nisp_blocks:
        if isinstance(block, (models.resnet.BasicBlock,models.resnet.Bottleneck)):
            print("Is residual block:  ",name)

            #TODO split into regular main path and residual path

            #TODO handle downsample path
        else:
            print("Not residual block: ",name)

            #TODO copy and edit from below loop over elementary modules
            #TODO add score pruning

def nisp_mag():
    #TODO apply nisp to FRL_mag scores
    pass

if __name__ == '__main__':
    from utils.pytorch_models import ResNet18
    model = ResNet18("r18", pretrained=False)
    nisp(model,frl_mag(model.FC, 0.3))
    print("--nisp call done--")

if __name__ == '__main__':
    from utils.pytorch_models import ResNet18
    model = ResNet18("r18", pretrained=False)

    # STATIC VARIANT: Collect ordered list of modules used in model
    used_modules_static = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            used_modules_static.append((name,module))
        elif isinstance(module, (nn.BatchNorm2d, nn.ReLU)):
            pass # not important for nisp
        elif len(list(module.children())) == 0:
            print("Elementary module for which there is no nisp implemented yet.. Check it out: ", name, module)
        else:
            print("Iteated over non elementary module:", name)
    print(20*"—","Used module list done.",20*"—")


    # Compute FRL Importance Scores by summing adjacent outgoing weight magnitudes
    FRL_activations = frl_mag(used_modules_static[-1][1],0.3)  # The layer befor size 19=#labels classification layer

    # First scores to be propagated back to next/previous layer
    reversed_importance_scores = [FRL_activations]



    for name, layer in reversed(used_modules_static[:-1]):
        #print(name)
        """
        #print("score shape: ", reversed_importance_scores[-1].shape)

        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            print("\n", layer._get_name(), "module weight shape: ", layer.weight.shape)
        else:
            print("\n", layer._get_name())

        #new_scores = nisp_importance()
        # prune.ln_structured(model.conv1, name="weight", amount=0.5, n=1, dim=0)

        # scores = prev_layer_scores * module.weight
        # prev_layer_scores = scores

        # print(module)
        # print(module(torch.randn(1, module.weight.shape[1], 32, 32)).shape)
        """

        S_out = reversed_importance_scores[-1]
        print("\nS_out.shape: ",S_out.shape,"     S_out sum: ", S_out.sum())#, S_out)

        if isinstance(layer, nn.Conv2d):
            print("nisped conv", name)
            # Option A: Use the conv_transpose2d approach for NISP
            reversed_importance_scores.append(
                nisp_conv2d(layer, S_out))
            # or the manual loop version, whichever you prefer

        elif isinstance(layer, nn.MaxPool2d):
            print("nisped max", name)
            # NISP paper does 'uniform distribution' for max-pool, so:
            reversed_importance_scores.append(
                nisp_maxpool2d(layer, S_out))
            # cannot rely on autograd alone, because it would only
            # backprop to the "winning" index

        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            print("nisped adapt", name)
            # Option A: use the manual bin-boundaries approach
            # Option B: use the autograd-based approach
            reversed_importance_scores.append(
                nisp_adaptiveavgpool2d_autograd(layer, S_out.view(512, 1, 1), (512, 10, 10)))
#"""

    #print("reversed importance scores",reversed_importance_scores)