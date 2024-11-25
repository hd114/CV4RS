import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path 

from functools import partial
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union
from typing import Container

from timm.models.convmixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer
from models.poolformer import PoolFormer
from utils.pytorch_models import ResNet50
import pandas as pd
from utils.BENv2_dataset import BENv2DataSet
from utils.pytorch_utils import (
    get_classification_report,
    init_results,
    print_micro_macro,
    update_results,
    start_cuda
)
from utils.pruning_utils import apply_pruning
from pxp import get_cnn_composite, ComponentAttribution
from pxp import GlobalPruningOperations


# data_dirs = {
#         "images_lmdb": "/data/kaiclasen/BENv2.lmdb",
#          "metadata_parquet": "/data/kaiclasen/metadata.parquet",
#          "metadata_snow_cloud_parquet": "/data/kaiclasen/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
#     }

# data_dirs = {
#         "images_lmdb": "Z:/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
#          "metadata_parquet": "Z:/faststorage/BigEarthNet-V2/metadata.parquet",
#          "metadata_snow_cloud_parquet": "Z:/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
#     }

data_dirs = {
        "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
         "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
         "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    }

#faststorage/BigEarthNet-V2.0


class PreFilter:
    def __init__(self, metadata: pd.DataFrame, countries: Optional[Container] | str = None, seasons: Optional[Container] | str = None):
        """
        Creates a function that filters patches based on country and season.

        Args:
            metadata: The metadata DataFrame.
            countries: A country or list of countries to include. If None, all countries are included.
            seasons: A season or list of seasons to include. If None, all seasons are included.
        """
        # add season info to metadata
        # months 12, 1, 2 are winter, 3, 4, 5 are spring, 6, 7, 8 are summer, 9, 10, 11 are autumn
        # get month based on patch_id (patch_id 15:17)
        metadata["month"] = metadata["patch_id"].str[15:17].astype(int)
        metadata["season"] = pd.cut(
            metadata["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Autumn"],
            right=False,
        )
        # manually set all entries with month 12 to winter
        metadata.loc[metadata["month"] == 12, "season"] = "Winter"

        seasons = None if seasons is None else seasons if isinstance(seasons, Container) else [seasons]
        countries = None if countries is None else countries if isinstance(countries, Container) else [countries]

        def filter_fn(metadata_row) -> bool:
            # Order: 'patch_id', 'labels', 'split', 'country', 's1_name', 's2v1_name',
            #        'contains_seasonal_snow', 'contains_cloud_or_shadow', 'month',
            #        'season'
            row_country = metadata_row[3]
            row_season = metadata_row[9]
            # check if patch season is correct
            if seasons is not None and row_season not in seasons:
                return False
            # check if patch country is correct
            if countries is not None and row_country not in countries:
                return False
            return True

        self.filter_fn = filter_fn
        from tqdm import tqdm
        self.filtered_patches = set([x[0] for x in [x for x in metadata.values if filter_fn(x)]])
        print(f"Pre-filtered {len(self.filtered_patches)} patches based on country and season (split ignored)")

    def filter(self, patch_id: str) -> bool:
        return self.filter_fn(patch_id)

    def __call__(self, patch_id: str) -> bool:
        return patch_id in self.filtered_patches


class Aggregator:
    def __init__(self) -> None:
        pass

    def fed_avg(self, model_updates: list[dict]) -> dict:
        """
        Performs federated averaging for model updates.

        Args:
            model_updates: A list of state dictionary differences from clients.

        Returns:
            dict: A state dictionary with the aggregated average.
        """
        assert len(model_updates) > 0, "No model updates provided."

        update_aggregation = {}
        for key in model_updates[0].keys():
            params = torch.stack([update[key] for update in model_updates], dim=0)
            avg = torch.mean(params, dim=0)
            update_aggregation[key] = avg

        return update_aggregation
class FLClient:
    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_path: list[str],
        batch_size: int = 256,
        num_workers: int = 2,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 19,
        device: torch.device = torch.device("cpu"),
        dataset_filter: str = "serbia",
        data_dirs: dict = None,
    ) -> None:
        """
        Initializes the Federated Learning Client.

        Args:
            model: PyTorch model instance.
            lmdb_path: Path to LMDB dataset.
            val_path: Path to validation dataset.
            csv_path: List of CSV files containing metadata.
            batch_size: Batch size for training and validation.
            num_workers: Number of workers for DataLoader.
            optimizer_constructor: Optimizer class.
            optimizer_kwargs: Optimizer configuration parameters.
            criterion_constructor: Loss function class.
            criterion_kwargs: Loss function configuration parameters.
            num_classes: Number of output classes.
            device: Device to use (CPU or GPU).
            dataset_filter: Filter criterion for dataset (e.g., country name).
            data_dirs: Dictionary containing dataset directories.
        """
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.results = init_results(self.num_classes)

        if data_dirs is None:
            raise ValueError("data_dirs cannot be None.")
        self.data_dirs = data_dirs

        self.train_loader = DataLoader(
            BENv2DataSet(
                data_dirs=self.data_dirs,
                split="train",
                img_size=(10, 120, 120),
                include_snowy=False,
                include_cloudy=False,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

        self.device = device

        self.val_loader = DataLoader(
            BENv2DataSet(
                data_dirs=self.data_dirs,
                split="test",
                img_size=(10, 120, 120),
                include_snowy=False,
                include_cloudy=False,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def train_one_round(self, epochs: int, validate: bool = False) -> dict:
        """
        Trains the model for one communication round.

        Args:
            epochs: Number of training epochs.
            validate: Whether to validate after training.

        Returns:
            dict: Model state dictionary updates.
        """
        state_before = copy.deepcopy(self.model.state_dict())
        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            self.train_epoch()

        if validate:
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)

        state_after = self.model.state_dict()
        model_update = {key: state_after[key] - state_before[key] for key in state_before.keys()}

        return model_update

    def train_epoch(self) -> None:
        """
        Trains the model for one epoch.
        """
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_loader, desc="training")):
            data = batch[1].to(self.device)
            labels = batch[4].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

    def validation_round(self) -> dict:
        """
        Evaluates the model on the validation dataset.

        Returns:
            dict: Validation metrics.
        """
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="test")):
                data = batch[1].to(self.device)
                labels = batch[4]
                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu()
                predicted_probs += list(probs.numpy())
                y_true += list(labels)

        predicted_probs = np.asarray(predicted_probs)
        y_true = np.asarray(y_true)

        report = get_classification_report(
            y_true, (predicted_probs >= 0.5).astype(float), predicted_probs, self.dataset_filter
        )
        return report


class Aggregator:
    def __init__(self) -> None:
        """
        Initializes the Aggregator for federated learning.
        """
        pass

    def fed_avg(self, model_updates: list[dict]) -> dict:
        """
        Implements Federated Averaging (FedAvg) to aggregate model updates.

        Args:
            model_updates: List of dictionaries containing model parameter updates.

        Returns:
            dict: Aggregated model parameter updates.
        """
        assert len(model_updates) > 0, "Trying to aggregate empty update list"

        update_aggregation = {}
        for key in model_updates[0].keys():
            params = torch.stack([update[key] for update in model_updates], dim=0)
            avg = torch.mean(params, dim=0)
            update_aggregation[key] = avg

        return update_aggregation



class Aggregator:
    def __init__(self) -> None:
        """
        Initializes the Aggregator for federated learning.
        """
        pass

    def fed_avg(self, model_updates: list[dict]) -> dict:
        """
        Implements Federated Averaging (FedAvg) to aggregate model updates.

        Args:
            model_updates: List of dictionaries containing model parameter updates.

        Returns:
            dict: Aggregated model parameter updates.
        """
        assert len(model_updates) > 0, "Trying to aggregate empty update list"

        update_aggregation = {}
        for key in model_updates[0].keys():
            params = torch.stack([update[key] for update in model_updates], dim=0)
            avg = torch.mean(params, dim=0)
            update_aggregation[key] = avg

        return update_aggregation


class GlobalClient:
    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_paths: list[str],
        batch_size: int = 128,
        num_workers: int = 0,
        num_classes: int = 19,
        dataset_filter: str = "serbia",
        state_dict_path: str = None,
        results_path: str = None,
        data_dirs: dict = None,
    ) -> None:
        """
        Initializes the GlobalClient responsible for managing clients and aggregation.

        Args:
            model: PyTorch model instance.
            lmdb_path: Path to LMDB dataset.
            val_path: Path to validation dataset.
            csv_paths: List of CSV paths for client data.
            batch_size: Batch size for training and validation.
            num_workers: Number of workers for DataLoader.
            num_classes: Number of output classes.
            dataset_filter: Filter criterion for dataset (e.g., country name).
            state_dict_path: Path to save model state dictionary.
            results_path: Path to save training results.
            data_dirs: Dictionary containing dataset directories.
        """
        if data_dirs is None:
            raise ValueError("data_dirs cannot be None. Please provide the necessary data directories.")

        self.data_dirs = data_dirs
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        self.model.to(self.device)

        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
        self.results = init_results(self.num_classes)

        # Initialize clients
        self.clients = [
            FLClient(
                copy.deepcopy(self.model),
                lmdb_path,
                val_path,
                csv_path,
                num_classes=num_classes,
                dataset_filter=dataset_filter,
                device=self.device,
                data_dirs=self.data_dirs,
            )
            for csv_path in csv_paths
        ]

        # Validation dataset
        self.countries = ["Finland", "Ireland", "Serbia"]
        self.validation_set = BENv2DataSet(
            data_dirs=self.data_dirs,
            split="test",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
        )

        # DataLoader for validation
        self.val_loader = DataLoader(
            self.validation_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

        # Paths for saving
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.state_dict_path = state_dict_path or f'checkpoints/global_model_{dt}.pkl'
        self.results_path = results_path or f'results/global_results_{dt}.pkl'

    def communication_round(self, epochs: int) -> None:
        """
        Executes a single communication round, including client training and aggregation.

        Args:
            epochs: Number of training epochs for each client.
        """
        model_updates = [client.train_one_round(epochs) for client in self.clients]

        # Aggregate parameter updates
        update_aggregation = self.aggregator.fed_avg(model_updates)

        # Update the global model
        global_state_dict = self.model.state_dict()
        for key, value in global_state_dict.items():
            update = update_aggregation[key].to(self.device)
            global_state_dict[key] = value + update
        self.model.load_state_dict(global_state_dict)

    def save_state_dict(self) -> None:
        """
        Saves the model's state dictionary to the specified path.
        """
        if not Path(self.state_dict_path).parent.is_dir():
            Path(self.state_dict_path).parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), self.state_dict_path)

    def save_results(self) -> None:
        """
        Saves the training results to the specified path.
        """
        if not Path(self.results_path).parent.is_dir():
            Path(self.results_path).parent.mkdir(parents=True)
        res = {'global': self.results, 'clients': [client.get_validation_results() for client in self.clients]}
        torch.save(res, self.results_path)
