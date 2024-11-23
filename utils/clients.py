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

    def fed_avg(self, model_updates: list[dict]):
        assert len(model_updates) > 0, "Trying to aggregate empty update list"
        
        update_aggregation = {}
        for key in model_updates[0].keys():
            params = torch.stack([update[key] for update in model_updates], dim=0)
            avg = torch.mean(params, dim=0)
            update_aggregation[key] = avg
        
        return update_aggregation


class FLCLient:
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
        device: torch.device = torch.device('cpu'),
        dataset_filter: str = "serbia",
    ) -> None:
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.pruning_mask = None  # Initialize pruning mask
        self.results = init_results(self.num_classes)
        self.dataset = BENv2DataSet(
        data_dirs=data_dirs,
        # For Mars use these paths
        split="train",
        img_size=(10, 120, 120),
        include_snowy=False,
        include_cloudy=False,
        patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=[csv_path], seasons=["Summer"]),
        )
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.device = device

        self.validation_set = BENv2DataSet(
        data_dirs=data_dirs,
        split="test",
        img_size=(10, 120, 120),
        include_snowy=False,
        include_cloudy=False,
        patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=[csv_path], seasons="Summer"),
        )
        self.val_loader = DataLoader(
            self.validation_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

    # def set_model(self, model: torch.nn.Module):
    #     self.model = copy.deepcopy(model)

    def set_model(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model)
        if self.pruning_mask is not None:
            # Apply pruning mask to the model
            state_dict = self.model.state_dict()
            for name, mask in self.pruning_mask.items():
                if name in state_dict:
                    state_dict[name] = state_dict[name] * mask  # Apply mask
            self.model.load_state_dict(state_dict)

    def train_one_round(self, epochs: int, validate: bool = False):
        state_before = copy.deepcopy(self.model.state_dict())

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        # criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)

            self.train_epoch()
        
        if validate:
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)

        state_after = self.model.state_dict()

        model_update = {}
        for key, value_before in state_before.items():
            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(
                torch.DoubleTensor
            )
            model_update[key] = diff

        return model_update

    def change_sizes(self, labels):
        new_labels=np.zeros((len(labels[0]),19))
        for i in range(len(labels[0])): #128
            for j in range(len(labels)): #19
                new_labels[i,j] =  int(labels[j][i])
        return new_labels
    
    def train_epoch(self):
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_loader, desc="training")):
            
        #    data, labels, index = batch["data"], batch["label"], batch["index"]
            data = batch[1]
            labels = batch[4]

            # print(f"Batch {idx}:")
            # print(f"  Data shape: {data.shape}")
            # print(f"  Labels shape: {labels.shape}")
            # print(f"  Sample label: {labels[0]}")
            # break  # Stop after the first batch for inspection
            
            data = data.cuda()
            label_new=np.copy(labels)
           # label_new=self.change_sizes(label_new)
            label_new = torch.from_numpy(label_new).cuda()
            self.optimizer.zero_grad()

            logits = self.model(data)
            # print(f"Logits sample: {logits[0]}")
            loss = self.criterion(logits, label_new)
            loss.backward()
            self.optimizer.step()
    
    
    
    def get_validation_results(self):
        return self.results

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
        results_path: str = None
    ) -> None:
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
        self.results = init_results(self.num_classes)
        self.clients = [
            FLCLient(copy.deepcopy(self.model), lmdb_path, val_path, csv_path, num_classes=num_classes, dataset_filter=dataset_filter, device=self.device)
            for csv_path in csv_paths
        ]
        self.validation_set = BENv2DataSet(
        data_dirs=data_dirs,
        split="test",
        img_size=(10, 120, 120),
        include_snowy=False,
        include_cloudy=False,
        patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=["Finland","Ireland","Serbia"], seasons="Summer"),
        )
        self.val_loader = DataLoader(
            self.validation_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
        
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if state_dict_path is None:
            if isinstance(model, ConvMixer):
                self.state_dict_path = f'checkpoints/global_convmixer_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.state_dict_path = f'checkpoints/global_mlpmixer_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.state_dict_path = f'checkpoints/global_poolformer_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.state_dict_path = f'checkpoints/global_resnet18_{dt}.pkl'

        if results_path is None:
            if isinstance(model, ConvMixer):
                self.results_path = f'results/convmixer_results_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.results_path = f'results/mlpmixer_results_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.results_path = f'results/poolformer_results_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.results_path = f'results/resnet18_results_{dt}.pkl'

    # def train(self, communication_rounds: int, epochs: int):
    #     start = time.perf_counter()
    #     for com_round in range(1, communication_rounds + 1):
    #         print("Round {}/{}".format(com_round, communication_rounds))
    #         print("-" * 10)
    #
    #         self.communication_round(epochs)
    #         report = self.validation_round()
    #
    #         self.results = update_results(self.results, report, self.num_classes)
    #         print_micro_macro(report)
    #
    #         for client in self.clients:
    #             client.set_model(self.model)
    #     self.train_time = time.perf_counter() - start
    #
    #     self.client_results = [client.get_validation_results() for client in self.clients]
    #     self.save_results()
    #     self.save_state_dict()
    #     return self.results, self.client_results

    def train(self, communication_rounds: int, epochs: int):
        start = time.perf_counter()
        pruning_ratio = 0.3  # Define pruning ratio
        global_pruning_mask = None  # Initialize pruning mask

        for com_round in range(1, communication_rounds + 1):
            print("Round {}/{}".format(com_round, communication_rounds))
            print("-" * 10)

            # Prüfen, ob die Maske existiert, und anwenden
            if global_pruning_mask is not None:
                state_dict = self.model.state_dict()
                for name, mask in global_pruning_mask.items():
                    if name in state_dict:
                        pruned_param = state_dict[name] * mask
                        # Überprüfen auf NaN-Werte nach dem Pruning
                        if torch.isnan(pruned_param).any():
                            print(f"NaN detected in layer {name} after pruning.")
                            raise ValueError("Pruned parameter contains NaN values.")
                        state_dict[name] = pruned_param  # Anwenden der Maske
                self.model.load_state_dict(state_dict)

            # Kommunikation und Training
            print(f"Training and communication for Round {com_round}...")
            self.communication_round(epochs)
            report = self.validation_round()

            # Ergebnisse aktualisieren und ausgeben
            self.results = update_results(self.results, report, self.num_classes)
            print_micro_macro(report)

            # Pruning nach der ersten Kommunikationsrunde anwenden
            if com_round == 1:
                print(f"Applying pruning after round {com_round}...")
                pruning_result = apply_pruning(self.model, pruning_ratio)
                global_pruning_mask = pruning_result["pruning_mask"]

                # Geprunte Gewichte laden
                pruned_state_dict = pruning_result["pruned_state_dict"]
                self.model.load_state_dict(pruned_state_dict)

                # Sicherstellen, dass die Maske direkt angewendet bleibt
                for name, mask in global_pruning_mask.items():
                    if name in pruned_state_dict:
                        pruned_param = pruned_state_dict[name] * mask
                        if torch.isnan(pruned_param).any():
                            print(f"NaN detected in layer {name} after pruning.")
                            raise ValueError("Pruned parameter contains NaN values.")
                        pruned_state_dict[name] = pruned_param
                self.model.load_state_dict(pruned_state_dict)

                # Maske an alle Clients senden
                for client in self.clients:
                    client.set_model(copy.deepcopy(self.model))
                    client.pruning_mask = global_pruning_mask

        # Abschluss der Trainingszeit
        self.train_time = time.perf_counter() - start

        # Ergebnisse der Clients sammeln
        self.client_results = [client.get_validation_results() for client in self.clients]
        self.save_results()
        self.save_state_dict()

        return self.results, self.client_results

    def change_sizes(self, labels):
        new_labels=np.zeros((len(labels[0]),19))
        for i in range(len(labels[0])): #128
            for j in range(len(labels)): #19
                new_labels[i,j] =  int(labels[j][i])
        return new_labels

    def validation_round(self):
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="test")):
                data = batch[1].to(self.device)
                labels = batch[4]

                label_new = np.copy(labels)
                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu()  # Wahrscheinlichkeiten berechnen

                # Variante 1: Threshold-basiert
                threshold = 0.5
                y_predicted = (probs >= threshold).float()  # Binäre Schwellenwert-basierte Vorhersage

                # Variante 2: Argmax-basiert
                # y_predicted = torch.zeros_like(probs)
                # y_predicted[torch.arange(probs.size(0)), probs.argmax(dim=1)] = 1  # Argmax-Vorhersage für eine Klasse

                predicted_probs += list(probs.numpy())  # Wahrscheinlichkeiten bleiben für andere Metriken erhalten
                y_true += list(label_new)

        predicted_probs = np.asarray(predicted_probs)
        y_true = np.asarray(y_true)

        # Ausgabe für Debugging
        # print(f"True labels shape: {y_true.shape}")
        # print(f"Predicted labels shape: {y_predicted.shape}")
        # print(f"Predicted probabilities shape: {predicted_probs.shape}")
        #
        # print(f"True labels sample: {y_true[:5]}")
        # print(f"Predicted labels sample: {y_predicted[:5]}")
        # print(f"Predicted probabilities sample: {predicted_probs[:5]}")

        # Überprüfen auf NaN-Werte in Arrays
        if np.isnan(predicted_probs).any():
            print("NaN detected in predicted probabilities array.")
            print(predicted_probs)
            raise ValueError("Predicted probabilities contain NaN values.")
        if np.isnan(y_true).any():
            print("NaN detected in true labels array.")
            print(y_true)
            raise ValueError("True labels contain NaN values.")

        report = get_classification_report(
            y_true, y_predicted.numpy(), predicted_probs, self.dataset_filter
        )
        return report

    def communication_round(self, epochs: int):
        # here the clients train
        # TODO: could be parallelized
        model_updates = [client.train_one_round(epochs) for client in self.clients]

        # parameter aggregation
        update_aggregation = self.aggregator.fed_avg(model_updates)

        # update the global model
        global_state_dict = self.model.state_dict()
        for key, value in global_state_dict.items():
            update = update_aggregation[key].to(self.device)
            global_state_dict[key] = value + update
        self.model.load_state_dict(global_state_dict)

    def save_state_dict(self):
        if not Path(self.state_dict_path).parent.is_dir():
            Path(self.state_dict_path).parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), self.state_dict_path)

    def save_results(self):
        if not Path(self.results_path).parent.is_dir():
            Path(self.results_path).parent.mkdir(parents=True)  
        res = {'global':self.results, 'clients':self.client_results, 'train_time': self.train_time}
        torch.save(res, self.results_path)
