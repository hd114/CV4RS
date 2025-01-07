import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data)


def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class ResNet18(nn.Module):
    def __init__(self, name, num_cls=19, channels=10, FC_dim=512, pretrained=None):
        super().__init__()
        self.name = name

        weights = None
        if pretrained:
            weights = ResNet18_Weights.DEFAULT

        resnet = models.resnet18(weights=weights)

        self.conv1 = nn.Conv2d(
            channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.FC = nn.Linear(FC_dim, num_cls)

        if not pretrained:
            self.apply(weights_init_kaiming)
            self.apply(fc_init_weights)

        # TODO ME necessary addition for adaptiveAvgPool nisp function backward trick
        self.adaptAvgPool_input_shape = -1
        def adaptAvgPool_input_shape_hook(adaptAvgPoolModule, inputs, outputs):
            # Forward hook that records shape of input to adaptiveAvgPool layer..needed for dummy_input in nisp function
            # `inputs` is a tuple; we want inputs[0].
            x = inputs[0]
            self.adaptAvgPool_input_shape = x.shape[1:] #ignores batch_size dimension
        hook_handle = resnet.avgpool.register_forward_hook(adaptAvgPool_input_shape_hook)

    #TODO ME list the ResidualBlocks
    def named_residualblocks(self):
        residual_blocks = []
        for name, module in self.named_modules():
            if isinstance(module, models.resnet.BasicBlock):
                residual_blocks.append([name,module])
        return residual_blocks

    def list_nisp_blocks(self):
        maxpool = self.encoder[3]
        avgpool = self.encoder[-1]
        blocks = (
                [["conv1",self.conv1], ["maxpool1",maxpool]]
                  + self.named_residualblocks()
                  + [["adaptavgpool1",avgpool]]
        )
        return blocks

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # Same as x.flatten(start_dim=1)
        logits = self.FC(x)
        return logits


class ResNet50(nn.Module):
    def __init__(self, name, num_cls=19, channels=10, FC_dim=2048, pretrained=False):
        super().__init__()
        self.name = name
        self.len = 0
        self.loss = 0

        weights = None
        if pretrained:
            weights = ResNet50_Weights.DEFAULT

        resnet = models.resnet50(weights=weights)

        self.conv1 = nn.Conv2d(
            channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(FC_dim, num_cls)

        if not pretrained:                          # previously functions were applied regardless of pretrained value
            self.apply(weights_init_kaiming)
            self.apply(fc_init_weights)

        #TODO ME necessary addition for adaptiveAvgPool nisp function backward trick
        self.adaptAvgPool_input_shape = -1
        def adaptAvgPool_input_shape_hook(adaptAvgPoolModule, inputs, outputs):
            # Forward hook that records shape of input to adaptiveAvgPool layer..needed for dummy_input in nisp function
            # `inputs` is a tuple; we want inputs[0].
            x = inputs[0]
            self.adaptAvgPool_input_shape = x.shape[1:] #ignores batch_size dimension
        hook_handle = resnet.avgpool.register_forward_hook(adaptAvgPool_input_shape_hook)

    # TODO ME list the ResidualBlocks
    def named_residualblocks(self):
        residual_blocks = []
        for name, module in self.named_modules():
            if isinstance(module, models.resnet.Bottleneck):
                residual_blocks.append([name,module])
        return residual_blocks

    def list_nisp_blocks(self):
        conv1_block = ["conv1",self.conv1]
        maxpool_block = ["maxpool1",self.encoder[3]]
        avgpool_block = ["adaptavgpool1",self.encoder[-1]]
        blocks = (
                [conv1_block, maxpool_block]
                  + self.named_residualblocks()
                  + [avgpool_block]
        )
        return blocks

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # Same as x.flatten(start_dim=1)
        logits = self.FC(x)
        return logits

class ViT_B_16(nn.Module):
    def __init__(self, name, num_cls=19, channels=10, pretrained=True):
        super().__init__()
        self.name = name

        # Load the pretrained vit_b_16 model
        self.vit = models.vit_b_16(pretrained=pretrained)

        # Replace the patch embedding to handle custom input channels
        self.input_conv = nn.Conv2d(
            channels, 3, kernel_size=1, stride=1, padding=0, bias=False
        )  # Converting to 3 channels expected by vit_b_16

        # Replace the classifier head for custom number of classes
        self.vit.heads = nn.Linear(self.vit.hidden_dim, num_cls)

    def forward(self, x):
        # Pass input through the custom input convolution
        x = self.input_conv(x)

        # Forward pass through ViT
        logits = self.vit(x)
        return logits

class ViT_B_16_alternative(nn.Module):
    def __init__(self, name, num_cls=19, channels=10, pretrained=False):
        super().__init__()
        self.name = name

        # Load VisionTransformer and customize the patch embedding
        self.vit = models.vision_transformer.VisionTransformer(
            image_size=224,#120 ?!
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=num_cls
        )

        # Replace the patch embedding layer to handle custom input channels
        self.vit.patch_embed = nn.Conv2d(
            channels, 768, kernel_size=16, stride=16, padding=0
        )

    def forward(self, x):
        # Forward pass through ViT
        logits = self.vit(x)
        return logits

if __name__ == '__main__':
    # How to use hooks provided by jonas https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging
    # dict(model.named_parameters()).keys() ._modules .named_modules     module.weight.data.shape  len(list(model.conv1.named_buffers())) model.conv1.named_parameters())["weight"].size()

    """

    print(10* "----PRUNE NOW------")
    #prune.random_unstructured(model.conv1, name="weight", amount=0.5)
            #  unstructured prunce single weights not rows/columns
    #prune.l1_unstructured(model.conv1, name="weight", amount=0.5)
            # HAS IMPORTANCE SCORES --> but unstructured prunce single weights not rows/columns, meaning not nodes
    #prune.random_structured(model.conv1, name="weight", amount=0.5, dim=1)
    #prune.ln_structured(model.conv1, name="weight", amount=0.5, n=1, dim=0) #IndexError: Invalid index 5 for tensor of size torch.Size([64, 10, 7, 7])
                # dim = 0    prune output channels (filters)
                # dim = 1   prune input channels per filter --> prunes same input channel for all filters for ln structured as well as random structured
                # dim = 2   filter height
                # dim = 3   filter width
                # For a convolutional layer, the weight tensor typically has the shape [out_channels, in_channels, height, width]
#TODO ln_structured pruning for the whole ResNet
    """
    """
    #print(10* "-----conv1 PARAMS AFTER PRUNE CONTAINS OLD WEIGHTS------")
    #print(list(model.conv1.named_parameters()))
    #print(10* "-----conv1 BUFFER AFTER PRUNE CONTAINS MASK------")
    #print(list(model.conv1.named_buffers()))
    #print(10* "-----conv1 PRUNED WEIGHTS------")
#    print(model.conv1)
#    print("model.conv1.weight.size():   ", model.conv1.weight.size())
#    torch.set_printoptions(threshold=100000)
#    print(model.conv1.weight)
#    print(model.conv1)
#    print("model.conv1.weight.size():   ", model.conv1.weight.size())
    #print(10* "-----con1 MASK------", "\n", model.conv1.weight_mask)
#    print(dict(model.conv1.named_parameters()).keys())
#    print(dict(model.conv1.named_buffers()).keys())

    #seq = dict(encoder.named_parameters()).keys()
    #print(dict(seq.named_parameters())['7.1.bn2.bias'])

    #print("shape ",torch.tensor(list(a.encoder.modules())).shape)
    #print(dict(seq.named_parameters())['7.1'])
    """
    """ TRACKING USED MODULES IN ORDER THROUGH HOOKS - NECESSIATEAS A FORWARD FEED
    used_modules = []
    def track_forward_hook(hooked_module, input, output):
        used_modules.append(hooked_module)

    # DYNAMIC VARIANT: Attach hooks to relevant layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.register_forward_hook(track_forward_hook)
    
    x = torch.randn(8, 10, 240, 240)
    output = model(x)
    
    print(used_modules)
    """

    print("ResNet18 names")
    model = ResNet18("r18", pretrained=False)
    for name, module in model.named_modules():
        if isinstance(module, models.resnet.BasicBlock):
            print(name)
    print(20 * "—", "Above are the ResNet18 ResidualBlocks called BasicBlock.", 20 * "—","\n")

    print("ResNet50 names")
    model50 = ResNet50("r50", pretrained=False)
    for name, module in model50.named_modules():
        if isinstance(module, models.resnet.Bottleneck):
            print(name)
    print(20 * "—", "Above are the ResNet50 ResidualBlocks called Bottleneck.", 20 * "—","\n")

    # STATIC VARIANT: Collect ordered list of modules used in model
    used_modules_static = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            used_modules_static.append(module)
        elif isinstance(module, (nn.BatchNorm2d, nn.ReLU)):
            pass  # not important for nisp
        elif len(list(module.children())) == 0:
            print("Seemingly an elementary module for which there is no nisp implemented yet.. Check it out: ", name,
                  module)
        else:
            print("Iteated over non elementary module:", name)
    print(20 * "—", "Used module list done for: ", model._get_name(), 20 * "—")

    # Compute FRL Importance Scores by summing adjacent outgoing weight magnitudes
    FRL_activations = torch.abs(used_modules_static[-1].weight).sum(dim=0) #Not for final layer with size 19=#labels but that before it

    # First scores to be propagated back to next/previous layer
    prev_layer_scores = FRL_activations

    for module in reversed(used_modules_static[:-1]):
        ##########print("\n",module._get_name(),"module weight shape: ", module.weight.shape)
        #print("score shape: ", prev_layer_scores.shape)
        #prune.ln_structured(model.conv1, name="weight", amount=0.5, n=1, dim=0)
        #scores = prev_layer_scores * module.weight
        #prev_layer_scores = scores
        print("-",module(torch.randn(1, 512, 10, 10)).shape)
        #print(module(torch.randn(1, 512, 10, 10)).shape)
        print(model.adaptAvgPool_input_shape)
        #print(module(torch.randn(1, module.weight.shape[1], 32, 32)).shape)

#######################################################################################
#######################################################################################
#######################################################################################
######################################################################################
    """
    # Dann feed samples to model

    for name, module in model.named_modules():
        print("ok", name, module)

        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)  # Prune filters
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)  # Prune nodes

        #BETTER USE HOOKS BECAUSE NAMED MODULES RETURNS THE WHOLE MODULE TREE NOT ONLY THE LEAFS
    """

"""
def forward_hook(module, input, output):
    prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)

# reversed(list(model.named_modules()))
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        module.register_forward_hook(forward_hook)

    # Forward pass
x = torch.randn(1, 1, 10, 10)
output = model(x)
"""