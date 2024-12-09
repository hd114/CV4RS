import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils.pytorch_models import ResNet50
from models.poolformer import create_poolformer_s12
from models.ConvMixer import create_convmixer
from models.MLPMixer import create_mlp_mixer
from utils.clients import GlobalClient
from utils.pytorch_utils import start_cuda


def ensure_model_keys(model: torch.nn.Module, required_keys: list[str]) -> torch.nn.Module:
    """
    Sicherstellen, dass alle Keys im Modell vorhanden sind.

    Args:
        model (torch.nn.Module): Das Modell.
        required_keys (list[str]): Liste der benötigten Keys.

    Returns:
        torch.nn.Module: Modell mit vollständigem State Dict.
    """
    state_dict = model.state_dict()
    for key in required_keys:
        if key not in state_dict:
            print(f"Initializing missing key: {key}")
            # Neuen Key als Null-Tensor initialisieren
            state_dict[key] = torch.zeros_like(next(iter(state_dict.values())))

    # Überarbeitetes State Dict laden
    model.load_state_dict(state_dict, strict=False)
    return model


def train():
	csv_paths = ["Finland","Ireland","Serbia"] #  ,"Ireland","Serbia"  this means that there are 3 clients that includes the images of a specific country. You can add Austria, Belgium, Lithuania, Portugal, Switzerland
	epochs = 3
	communication_rounds = 8
	channels = 10
	num_classes = 19
	#model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
	#model = create_mlp_mixer(channels, num_classes)
	#model = create_convmixer(channels=channels, num_classes=num_classes, pretrained=False)
    #model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
	model = ResNet50("ResNet50", channels=channels, num_cls=num_classes, pretrained=False)
 
	# Sicherstellen, dass alle Keys vorhanden sind
	required_keys = [
        "conv1.bias",
        "encoder.1.bias",
        # Weitere Keys aus Debugging-Informationen hinzufügen...
    ]
	model = ensure_model_keys(model, required_keys)

    # GlobalClient instanziieren
	global_client = GlobalClient(
        model=model,
        lmdb_path="",
        val_path="",
        csv_paths=csv_paths,
    )
	global_model, global_results = global_client.train(communication_rounds=communication_rounds, epochs=epochs)
	print(global_results)

if __name__ == '__main__':
    train()