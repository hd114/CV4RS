import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils.pytorch_models import ResNet50
from models.poolformer import create_poolformer_s12
from models.ConvMixer import create_convmixer
from models.MLPMixer import create_mlp_mixer
from utils.clients import GlobalClient
from utils.pytorch_utils import start_cuda


def train():
    csv_paths = ["Finland"]  # Liste der Länder für die Clients
    epochs = 1
    communication_rounds = 3
    channels = 10
    num_classes = 19

    # Modell initialisieren (ResNet50 in diesem Fall)
    model = ResNet50("ResNet50", channels=channels, num_cls=num_classes, pretrained=False)

    # Datenverzeichnisse definieren
    data_dirs = {
        "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
        "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
        "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    }

    # GlobalClient instanziieren
    global_client = GlobalClient(
        model=model,
        lmdb_path="",
        val_path="",
        csv_paths=csv_paths,
        data_dirs=data_dirs,  # Hier wird `data_dirs` übergeben
    )

    # Training starten
    global_model, global_results = global_client.train(
        communication_rounds=communication_rounds,
        epochs=epochs
    )
    print(global_results)


if __name__ == '__main__':
    train()