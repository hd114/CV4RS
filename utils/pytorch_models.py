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

        self.FC_dim = FC_dim #Facilitates reshaping the FRL_scores for the adaptAvgPool module nisp
        # TODO ME necessary addition for adaptiveAvgPool nisp function backward trick
        self.adaptAvgPool_input_shape = [512, 4, 4] # initialization value based on patch shape (10, 120, 120)
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
        conv1_block = ["conv1", self.conv1]
        maxpool_block = ["encoder.3", self.encoder[3]]
        avgpool_block = ["encoder.8", self.encoder[-1]]
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


        self.FC_dim = FC_dim #Facilitates reshaping the FRL_scores for the adaptAvgPool module nisp
        #TODO ME necessary addition for adaptiveAvgPool nisp function backward trick
        self.adaptAvgPool_input_shape = [2048, 4, 4] # initialization value based on patch shape (10, 120, 120)
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
        maxpool_block = ["encoder.3",self.encoder[3]]
        avgpool_block = ["encoder.8",self.encoder[-1]]
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