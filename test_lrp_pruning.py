import torch
from pxp import get_cnn_composite, ComponentAttribution

# Modell initialisieren
model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
model.eval()

# Dummy-Daten erstellen (nur Bilder, keine Labels)
# Dummy-Daten erstellen (Bilder + Labels)
class DummyDataloader:
    def __iter__(self):
        for _ in range(10):  # 10 Iterationen simulieren
            yield torch.randn(1, 3, 224, 224), torch.tensor([0])  # Bild + Dummy-Label


# Dummy-Dataloader initialisieren
dataloader = DummyDataloader()

# LRP-Pruning-Setup
composite = get_cnn_composite("resnet50", {
    "low_level_hidden_layer_rule": "Epsilon",
    "mid_level_hidden_layer_rule": "Epsilon",
    "high_level_hidden_layer_rule": "Epsilon",
    "fully_connected_layers_rule": "Epsilon",
    "softmax_rule": "Epsilon",
})
component_attributor = ComponentAttribution("Relevance", "CNN", torch.nn.Conv2d)

# Relevanzen berechnen
for images in dataloader:
    relevance = component_attributor.attribute(
        model=model,
        dataloader=[images],  # Nur Bilder übergeben
        attribution_composite=composite,
        abs_flag=True,
        device="cpu",
    )
    print(f"Relevance computed: {relevance}")

