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

# Example instantiation
# model = ViT_B_16(name="ViT_B_16_Custom", num_cls=19, channels=10, pretrained=True)
# x = torch.randn(8, 10, 224, 224)  # Batch of 8, 10 channels, 224x224 images
# logits = model(x)
# print(logits.shape)

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

# Example instantiation
# model = ViT_B_16_alternative(name="ViT_B_16_Custom", num_cls=19, channels=10, pretrained=False)
# x = torch.randn(8, 10, 224, 224)  # Batch of 8, 10 channels, 224x224 images
# logits = model(x)
# print(logits.shape)



if __name__ == '__main__':
    a = ResNet18("r18")

    #print(50*"#")
    #print(models.vit_b_16(pretrained=False))
    #print(50 * "#")
    #print(ResNet50("r50"))
    #print(res18.encoder)
    #print(models.resnet18(pretrained=True).modules())
    #print(models.resnet18(pretrained=True).get_submodule())

    module = a.conv1
    print("type(module):    ", type(module))
    #print(module.weight.data)
    print(module.weight.data.shape) # -> torch.Size([64, 10, 7, 7])
    print(list(module.named_parameters()))

    print(a.encoder.modules()[0])