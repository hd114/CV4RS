import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.pytorch_models import ResNet18, ResNet50

def output_padding_based_on_stride(stride_to_deconvolve):
    if isinstance(stride_to_deconvolve, int):
        return stride_to_deconvolve-1
    else:
        return stride_to_deconvolve[0]-1, stride_to_deconvolve[1]-1


def nisp_Conv2d(conv_module: nn.Conv2d, scores_to_propagate: torch.Tensor, pause_output_padding: bool) -> torch.Tensor:
    """
    Propagate convolution output layer importance scores(scores_to_propagate) back to convolution input layer importance scores for a
    given Conv2d module (conv_module). Handling arbitrary stride, padding, and dilation via transposed convolution.

    Args:
        conv_module (nn.Conv2d):
            Convolution module over which importance scores are to be propagated back.
        scores_to_propagate (torch.Tensor):
            Importance scores of shape (C_out, H_out, W_out) to propagate back over convolution module.

    Returns:
        S_in (torch.Tensor):
            Convolution input layer importance scores of shape (C_in, H_in, W_in), where H_in and W_in are inferred/resulting from
            conv parameters and scores_to_propagate size.
    """
    weight_used = torch.abs(conv_module.weight)  # without transposing weights here!!
    #print("Weight used shape:", weight_used.shape)
    #print("Input scores shape:", scores.shape)
    #assert(weight_used.shape[0] == scores_to_propagate.shape[0]) #TODO recheck

    # add batch dimension
    scores_batched = scores_to_propagate.detach().clone().unsqueeze(0)  # Now shape [1, 512, 4, 4]

    S_in_4d = F.conv_transpose2d(
        scores_batched,
        weight_used,
        bias=None,
        stride=conv_module.stride,
        padding=conv_module.padding,
        output_padding=0 if pause_output_padding else output_padding_based_on_stride(conv_module.stride)
    )
    return S_in_4d.squeeze(0)

def nisp_MaxPool2d(pool_module: nn.MaxPool2d, scores_to_propagate: torch.Tensor, pause_output_padding: bool) -> torch.Tensor:
    """
    NISP for convolution module(pool_module) using a grouped transposed convolution with same kernel_size, stride,
    padding, and dilation to uniformly distribute output importance scores(scores_to_propagate) across each pooling region.
    (Not only the pooled activations receive scores!! pooled and unpooled treated equally!!!)

    Args:
        pool_module (nn.MaxPool2d):
            Pooling module  over which importance scores are to be propagated back.
        scores_to_propagate (torch.Tensor):
           Importance scores of shape (C, H_out, W_out) to propagate back over max pooling module.

    Returns:
        S_in (torch.Tensor):
            Pooling input layer importance scores of shape (C, H_in, W_in) resulting from score propagation.
    """

    # PHASE 1) Extract pooling parameters (kernel_size, stride, padding, dilation) for transposed convolution
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

    C, H_out, W_out = scores_to_propagate.shape

    # PHASE 2) Build transposed-conv kernel of all ones, shaped (C, 1, kH, kW),
    #    and set groups=C so each channel is handled separately.
    weight_ones = torch.ones(                                                                  # torch.abs(ones) == ones
        (C, 1, kH, kW),
        dtype=scores_to_propagate.dtype,
        device=scores_to_propagate.device
    )

    # We treat S_out as (N=1, C, H_out, W_out) for the transposed convolution.
    S_out_4d = scores_to_propagate.unsqueeze(0)

    #PHASE 3) Perform grouped transposed convolution:
    #    - No bias
    #    - groups=C  ensures per-channel “stamping”
    S_in_4d = F.conv_transpose2d(
        S_out_4d,
        weight_ones,
        bias=None,
        stride=(sH, sW),
        padding=(pH, pW),
        output_padding=0 if pause_output_padding else output_padding_based_on_stride(pool_module.stride),
        groups=C,
        dilation=(dH, dW)
    )
    # S_in_4d now has shape (1, C, H_in, W_in), but each (kH,kW) block is a sum of S_out.
    # We want a uniform distribution, so we divide by (kH*kW).
    S_in_4d /= (kH * kW)                                                                        #Actually can be ignored

    # Remove the batch dimension -> (C, H_in, W_in)
    S_in = S_in_4d.squeeze(0)

    return S_in

def nisp_AdaptiveAvgPool2d_autograd(
    pool_module: nn.AdaptiveAvgPool2d,
    scores_to_propagate: torch.Tensor,
    input_shape: tuple
) -> torch.Tensor:
    """
    Propagate importance scores back over AdaptiveAvgPool2d(pool_module) by uniformly distributing pooling output
    importance scores(scores_to_propagate) across adaptively pooled regions. Implemented through torch autograd trick on constructed
    dummy input with shape(input_shape) of adaptive pooling input. Resulting .grad is pooling input layer importance.

    Args:
        pool_module (nn.AdaptiveAvgPool2d):
            The adaptive pooling module whose backward pass we want to mimic.
        scores_to_propagate (torch.Tensor):
            Output importance of shape (C, H_out, W_out) to propagate over adaptive average pooling module.
        input_shape (tuple):
            Shape of original input (C, H_in, W_in) that was fed turing training/inference into module.
            Custom ResNets provide this through adaptAvgPool_input_shape_hook.

    Returns:
        S_in (torch.Tensor):
            Pooling input layer importance scores of shape (C, H_in, W_in) resulting from score propagation.
    """
    C, H_in, W_in = input_shape
    C_out, H_out, W_out = scores_to_propagate.shape
    assert C == C_out, "Channel mismatch between scores_to_propagate and input_shape"

    # 1) Create a dummy input with requires_grad=True
    #    We'll add a batch dimension of 1 => (1, C, H_in, W_in).
    dummy_input = torch.zeros(
        (1, C, H_in, W_in),
        dtype=scores_to_propagate.dtype,
        device=scores_to_propagate.device,
        requires_grad=True
    )

    # 2) Forward pass through the AdaptiveAvgPool2d
    out = pool_module(dummy_input)  # shape => (1, C, H_out, W_out)

                                                                        # BACKWARD OVER AVGPOOL SHOULD NOT CHANGE SIGN
    # 3) Backward pass: treat scores_to_propagate as the "gradient" from above
    out.backward(scores_to_propagate.detach().clone().unsqueeze(0))  # scores_to_propagate => (C, H_out, W_out), unsqueeze => (1, C, H_out, W_out)

    # 4) The gradient w.r.t. dummy_input is exactly our importance
    S_in = dummy_input.grad.detach().squeeze(0)  # => shape (C, H_in, W_in)

    return S_in

def zero_out_smallest_scores(importance_scores: torch.Tensor, pruning_rate: float) -> torch.Tensor:
    """
    Zeroes out the smallest importance scores at rate pruning_rate.

    Args:
        importance_scores (torch.Tensor): Scores tensor.
        pruning_rate (float): Rate of zeroing out small scores (0.0-0.99).

    Returns:
        torch.Tensor: Modified tensor with the smallest elements zeroed out.
    """
    scores_flattened = importance_scores.detach().clone().flatten()
    k = int(len(scores_flattened) * pruning_rate)  # Number of elements to zero out

    if k > 0:
        threshold = torch.kthvalue(scores_flattened, k).values  # Get the k-th smallest value
        mask = scores_flattened >= threshold  # Mask: Keep only values >= threshold #TODO check scores distibution and decide between >= and >
        scores_flattened = scores_flattened * mask  # Zero out smallest values

    return scores_flattened.view_as(importance_scores)


def zero_out_smallest_channels(importance_scores: torch.Tensor, pruning_rate: float) -> torch.Tensor:
    """
    Zeros out the channels with the smallest average value.

    Args:
        importance_scores (torch.Tensor): Scores tensor of shape (C, H, W).
        pruning_rate (float): Rate of zeroing out small value channels (0.0-0.99).

    Returns:
        torch.Tensor: Score tensor of same shape as input tensor, but with the smallest value channels set to zero.
    """
    # 1. Compute mean across (H, W) for each channel
    #    channel_means will be of shape (C,)
    channel_means = importance_scores.mean(dim=(1, 2))

    k = int(len(channel_means) * pruning_rate)

    # 2. Identify indices of the k channels with the smallest average
    #    largest=False makes .topk() return the smallest values
    _, smallest_k_indices = torch.topk(channel_means, k, largest=False)

    # 3. Clone x to avoid changing original tensor in-place
    pruned_scores = importance_scores.detach().clone()

    # 4. Zero out the channels identified by smallest_k_indices
    #    smallest_k_indices is 1D, so we use it to index the channel dimension
    pruned_scores[smallest_k_indices, :, :] = 0

    return pruned_scores

def frl_mag(final_module: nn.Linear) -> torch.Tensor:
    return torch.abs(final_module.weight).sum(dim=0)

def inf_fs(final_module: nn.Linear) -> torch.Tensor:
    pass

def nisp_leaf_module(custom_resnet: nn.Module, leaf_name: str, leaf_module: nn.Module, importance_scores: torch.Tensor, pruning_rate: float, pause_output_padding: bool, protected_modules: list[str]):

    if isinstance(leaf_module, nn.Conv2d):
        print(leaf_name+".weight.shape: ", leaf_module.weight.shape)
        output_scores = nisp_Conv2d(leaf_module, importance_scores,pause_output_padding)
    elif isinstance(leaf_module, nn.MaxPool2d):
        output_scores = nisp_MaxPool2d(leaf_module, importance_scores,pause_output_padding)
    elif isinstance(leaf_module, nn.AdaptiveAvgPool2d):
        output_scores = nisp_AdaptiveAvgPool2d_autograd(leaf_module, importance_scores, custom_resnet.adaptAvgPool_input_shape)
    else:
        raise NotImplementedError(f"Block type {type(leaf_module)} not handled.")

    if leaf_name not in protected_modules:
        output_scores = zero_out_smallest_channels(output_scores, pruning_rate)

    return output_scores

def scores_to_weight_mask(module_to_mask: nn.Module ,scores_to_propagate: torch.Tensor) -> torch.Tensor:
    return (scores_to_propagate.detach().clone() > 0).float().mean(dim=(1, 2)).view(-1,1,1,1).expand_as(module_to_mask.weight)

def nisp(custom_resnet: nn.Module, FRL_scores: torch.Tensor, pruning_rate: float, protected_modules: list[str]): # not needed anymore -> , dummy_image_size=(1000,1,1)):
    """
    Propagate provided FRL_scores over custom ResNet(expects same interface as utils/pytorch_models.py imlementations) by
    splitting it into blocks and applying nisp per block.
    See Table 1 on page 5: https://arxiv.org/abs/1512.03385
    Residual blocks are treated by performing nisp in parallel for the main path as well as the skip connection path.
    The skip connection path only contains the (downsample) composed module.
    THe main bath consists of multiple convolutions

    Args:
        custom_resnet (nn.Module):
            The custom resnet over which to apply nisp.
        FRL_scores (torch.Tensor):
            The importance scores of shape (C, H_out, W_out) of the final response layer, the layer before classification.
        pruning_rate (float):
            The rate at which to zero out importance scores per layer which leads to them being pruned later.

    Returns:
        S_in (torch.Tensor):
            The propagated importance, shape (C, H_in, W_in).
    """

    if isinstance(custom_resnet,(ResNet18,ResNet50)):
        hardcoded_block_pause_outputpadding = "encoder.6.0"
    else:
        raise NotImplementedError("first implement mechanism to choose for your architecture correct output_padding in transposed convolutions!")

    nisp_blocks = custom_resnet.list_nisp_blocks() # encoder.named_children() is not sufficient!

    mask = None #TODO create based on provided LRP pruning

    pruned_FRL_scores_flat = zero_out_smallest_scores(FRL_scores, pruning_rate)

    scores_to_propagate = pruned_FRL_scores_flat.view(custom_resnet.FC_dim, 1, 1) # resnet18 final conv has 512 channels, resnet50 has 2048

    scores_dict = {}#"FC":pruned_FRL_scores_flat.detach().clone()}

    for name, block in reversed(nisp_blocks):
        print("\nscores_to_propagate.shape: ",scores_to_propagate.shape)

        pause_output_padding = name == hardcoded_block_pause_outputpadding

        if isinstance(block, (models.resnet.BasicBlock,models.resnet.Bottleneck)):
            ##print("Is residual block:  ",name)

            if block.downsample:
                ##print(name + ".downsample.0")
                scores_dict[name + ".downsample.0"] = scores_to_weight_mask(block.downsample._modules["0"],scores_to_propagate) #TODO think about why we assign scores to module before nisping over it; reason is that we use scores for >>OUTPUT<< neuron/channel/filter pruning
                                                                                            #otherwise it would also be residual_scores.detach().clone() instead of scores_to_propagate.detach().clone()
                residual_scores = nisp_leaf_module(custom_resnet,name+".downsample.0", block.downsample._modules["0"], scores_to_propagate, pruning_rate,pause_output_padding=pause_output_padding,protected_modules=protected_modules)
            #TODO think about this issue: between layers importance score magnitudes can differ.
            # Skip connections skip over multiple modules, importance score then have different magnitude scales than main path scores.. normalize per module ? not necessary except if we later want non-layerbounded pruning

            for main_path_module_name, main_path_module in reversed(list(block.named_children())): # named_children instead of modules so that the deeper nested "downsample.0" convolution is skipped over
                if isinstance(main_path_module, nn.Conv2d):
                    ##print(name+"."+main_path_module_name)
                    scores_dict[name+"."+main_path_module_name] = scores_to_weight_mask(main_path_module,scores_to_propagate)
                    scores_to_propagate = nisp_leaf_module(custom_resnet, name+"."+main_path_module_name, main_path_module, scores_to_propagate, pruning_rate,pause_output_padding=pause_output_padding,protected_modules=protected_modules)

            if block.downsample:
                #assert(torch.equal(scores_to_propagate.shape, residual_scores.shape))
                scores_to_propagate += residual_scores # skip and main path scores are merged, then propagated;this increases leads to shallow layers automatically having bigger scores than
        else:
            ##print("leaf module",name,type(block))
            if isinstance(block, (nn.Conv2d, nn.Linear)) and (name not in protected_modules):
                scores_dict[name] = scores_to_weight_mask(block,scores_to_propagate)
            scores_to_propagate = nisp_leaf_module(custom_resnet, name, block, scores_to_propagate, pruning_rate,pause_output_padding=pause_output_padding, protected_modules=protected_modules)

    print("\nInput layer scores: ", scores_to_propagate.shape)
    #print(scores_to_propagate)

    return scores_dict

def nisp_mag(resnet: nn.Module,pruning_rate: float, protected_modules: list[str]):
    return nisp(resnet, frl_mag(resnet.FC),pruning_rate,[])

if __name__ == '__main__':
    model = ResNet18("r18", pretrained=True) # pretrained=False)
    model(torch.randn(1,10,120,120))

    scores = nisp_mag(model, pruning_rate=0.5,protected_modules=["FC"])
    print(scores.keys())
    print(model.conv1.weight.shape)
    for key in scores.keys():
        print(scores[key].shape)
    #print(scores["conv1"].mean(dim=(1, 2)).view(-1,1,1,1).expand_as(model.conv1.weight).shape)#.max(dim=1,2).values.shape)