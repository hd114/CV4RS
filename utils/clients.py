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

from pyinstrument import Profiler
import sys
import os
from pyinstrument import Profiler

import cProfile

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
        device: torch.device = torch.device("cpu"),
        dataset_filter: str = "serbia",
        data_dirs: dict = None,  # Neu hinzugefügt
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

        # Speichern von data_dirs
        if data_dirs is None:
            raise ValueError("data_dirs cannot be None.")
        self.data_dirs = data_dirs

        self.dataset = BENv2DataSet(
            data_dirs=self.data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(
                pd.read_parquet(self.data_dirs["metadata_parquet"]),
                countries=[csv_path],
                seasons=["Summer"],
            ),
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
            data_dirs=self.data_dirs,
            split="test",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(
                pd.read_parquet(self.data_dirs["metadata_parquet"]),
                countries=[csv_path],
                seasons="Summer",
            ),
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

            # Debugging: Ausgabe der Batch-Dimensionen
            '''print(f"Batch {idx}:")
            print(f"  Data shape: {data.shape}")  # Eingabedatenform prüfen
            print(f"  Labels shape: {labels.shape}")  # Label-Form prüfen

            # Debugging: Conv1-Gewichte überprüfen
            print(f"  Conv1 weight shape: {self.model.encoder[0].weight.shape}")'''
            
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
            results_path: str = None,
            data_dirs: dict = None,
    ) -> None:
        # Überprüfen, ob data_dirs definiert ist
        if data_dirs is None:
            raise ValueError("data_dirs cannot be None. Please provide the necessary data directories.")

        # Daten speichern
        self.data_dirs = data_dirs
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        self.model.to(self.device)

        # Initialisierungen
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
        self.results = init_results(self.num_classes)

        # Clients initialisieren
        self.clients = [
            FLCLient(
                copy.deepcopy(self.model),
                lmdb_path,
                val_path,
                csv_path,
                num_classes=num_classes,
                dataset_filter=dataset_filter,
                device=self.device,
                data_dirs=self.data_dirs,  # Datenverzeichnisse weitergeben
            )
            for csv_path in csv_paths
        ]

        # Validierungs-Dataset
        self.countries = ["Finland", "Ireland", "Serbia"]  # Standard-Länder
        self.validation_set = BENv2DataSet(
            data_dirs=self.data_dirs,
            split="test",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(
                pd.read_parquet(self.data_dirs["metadata_parquet"]),
                countries=self.countries,  # Länderfilter dynamisch
                seasons="Summer",
            ),
        )

        # DataLoader für Validierung
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

    def get_country_dataloader(self, country: str, batch_size: int, num_workers: int):
        """
        Erstellt einen DataLoader, der nur Daten eines bestimmten Landes enthält.

        Args:
            country (str): Das gewünschte Land (z. B. "Finland").
            batch_size (int): Batch-Größe für den DataLoader.
            num_workers (int): Anzahl der Worker-Threads.

        Returns:
            DataLoader: Ein DataLoader, der Daten aus dem spezifischen Land enthält.
        """
        print(f"Erstelle DataLoader für Land: {country}")
        filtered_dataset = BENv2DataSet(
            data_dirs=self.data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(
                pd.read_parquet(self.data_dirs["metadata_parquet"]),
                countries=[country],  # Filter auf das gewünschte Land
                seasons="Summer",
            ),
        )
        return DataLoader(
            filtered_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def compute_lrp_pruning_mask(self, composite, component_attributor, pruning_rate=0.3):
        """
        Compute the LRP pruning mask based on global relevance maps.

        Args:
            composite: The LRP composite object.
            component_attributor: The component attribution object.
            pruning_rate (float): Fraction of filters to prune.

        Returns:
            dict: The global pruning mask.
        """
        print("[INFO] Computing LRP pruning mask...")
        dataloader = self.get_country_dataloader("Finland", batch_size=16, num_workers=4)

        # Compute relevance maps
        global_concept_maps = component_attributor.attribute(
            model=self.model,
            dataloader=((batch[0].float(), batch[-1]) for batch in dataloader),
            attribution_composite=composite,
            abs_flag=True,
            device=self.device,
        )
        print(f"[INFO] Relevance maps computed for {len(global_concept_maps)} layers.")

        # Debugging: Check the generated mask
        for layer_name, relevance_map in global_concept_maps.items():
            print(f"[DEBUG] Layer: {layer_name}")
            if isinstance(relevance_map, dict):
                for mask_type, mask in relevance_map.items():
                    print(
                        f"  Mask type: {mask_type}, Mask shape: {mask.shape}, Non-zero elements: {torch.sum(mask != 0)}")
            else:
                print(f"  Mask shape: {relevance_map.shape}, Non-zero elements: {torch.sum(relevance_map != 0)}")

        # Generate global pruning mask
        pruning_ops = GlobalPruningOperations(
            target_layer=torch.nn.Conv2d,
            layer_names=list(global_concept_maps.keys()),
        )
        global_pruning_mask = pruning_ops.generate_global_pruning_mask(
            model=self.model,
            global_concept_maps=global_concept_maps,
            pruning_percentage=pruning_rate,
            least_relevant_first=True,
            device=self.device,
        )

        return global_pruning_mask

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
        """
        Train the global model across multiple communication rounds with pruning and validation.

        Args:
            communication_rounds (int): Number of communication rounds.
            epochs (int): Number of epochs per communication round.

        Returns:
            tuple: (global results, client results)
        """
        start = time.perf_counter()
        pruning_rate = 0.3  # Pruning rate
        global_pruning_mask = None  # Initialize pruning mask
        pruning_ops = None  # Ensure pruning_ops is initialized outside the loop

        # LRP-Pruning initialization
        print("Initializing LRP Pruning...")
        composite, component_attributor = self.initialize_lrp_pruning("resnet50", torch.nn.Conv2d)
        print("LRP initialized successfully.")

        for com_round in range(1, communication_rounds + 1):
            print(f"=== Round {com_round}/{communication_rounds} ===")

            # Apply pruning mask if available
            if global_pruning_mask is not None:
                print(f"Applying pruning mask for Round {com_round}...")
                for layer_name, layer_pruning_mask in global_pruning_mask.items():
                    print(f"[DEBUG] Validating mask for layer: {layer_name}")
                    for mask_type, mask in layer_pruning_mask.items():
                        if isinstance(mask, torch.Tensor):  # Überprüfen, ob es sich um einen Tensor handelt
                            print(
                                f"  Mask type: {mask_type}, Mask shape: {mask.shape}, Non-zero elements: {torch.sum(mask != 0)}"
                            )
                        else:
                            print(f"[WARNING] Mask for {mask_type} is not a Tensor: {type(mask)}")

                    print(f"[INFO] Applying pruning mask to layer: {layer_name}")
                    try:
                        pruning_ops.fit_pruning_mask(self.model, layer_name, layer_pruning_mask)
                    except Exception as e:
                        print(f"[ERROR] Failed to apply pruning mask to layer {layer_name}: {e}")
                        raise

                # Validate pruning
                for name, param in self.model.named_parameters():
                    if name in global_pruning_mask:
                        print(f"[DEBUG] Layer {name}: Non-zero weights after pruning: {torch.sum(param != 0)}")

            # Training and communication
            print(f"Training and communication for Round {com_round}...")
            self.communication_round(epochs)

            # Perform validation after each communication round
            print(f"[INFO] Starting validation for Round {com_round}...")
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)
            print_micro_macro(report)

            # Update client models
            for client in self.clients:
                client.set_model(self.model)

            # Perform LRP pruning after the first communication round
            if com_round == 1:
                print(f"[INFO] Performing LRP Pruning in Round {com_round}...")
                profiler = Profiler()
                profiler.start()

                try:
                    global_concept_maps = self.compute_lrp_pruning_mask(
                        composite=composite,
                        component_attributor=component_attributor,
                        pruning_rate=pruning_rate,
                    )
                    print(f"[DEBUG] Global concept maps computed with {len(global_concept_maps)} layers.")

                    # Initialize pruning operations
                    pruning_ops = GlobalPruningOperations(
                        target_layer=torch.nn.Conv2d,
                        layer_names=list(global_concept_maps.keys())
                    )

                    # Generate global pruning mask
                    global_pruning_mask = pruning_ops.generate_global_pruning_mask(
                        model=self.model,
                        global_concept_maps=global_concept_maps,
                        pruning_percentage=pruning_rate,
                        least_relevant_first=True,
                        device=self.device,
                    )
                    print(f"[DEBUG] Global pruning mask: {global_pruning_mask}")

                except Exception as e:
                    print(f"[ERROR] Failed to compute pruning mask: {e}")
                    raise

                profiler.stop()
                try:
                    text_file_name = "pruning_callgraph.txt"
                    with open(text_file_name, "w", encoding="utf-8") as text_file:
                        full_text_output = profiler.output_text(unicode=True, color=False, show_all=True)
                        text_file.write(full_text_output)
                    print(f"Full profiling results saved to {os.path.abspath(text_file_name)}")
                except Exception as e:
                    print(f"Error while writing profiling results: {e}")

        # Finalize training time
        self.train_time = time.perf_counter() - start
        print(f"Training completed in {self.train_time:.2f} seconds.")

        # Collect client results
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
        #print("[DEBUG] validation_round called")
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
        print(f"True labels sample: {y_true[:5]}")
        print(f"Predicted labels sample: {y_predicted[:5]}")
        print(f"Predicted probabilities sample: {predicted_probs[:5]}")

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
        # Clients trainieren
        model_updates = [client.train_one_round(epochs) for client in self.clients]

        # Parameteraggregation
        update_aggregation = self.aggregator.fed_avg(model_updates)

        # Globales Modell aktualisieren
        global_state_dict = self.model.state_dict()
        for key, value in global_state_dict.items():
            if key in update_aggregation:
                update = update_aggregation[key].to(self.device)
                global_state_dict[key] = value + update
            else:
                print(f"Skipping missing parameter: {key}")
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

    def initialize_lrp_pruning(self, model_name, layer_type):
        """
        Initialisiert LRP-Pruning-Komponenten.

        Args:
            model_name (str): Der Name des Modells (z. B. "resnet50").
            layer_type: Der Layer-Typ, der für das Pruning verwendet wird (z. B. torch.nn.Conv2d).

        Returns:
            tuple: Ein Composite-Objekt und ein ComponentAttribution-Objekt.
        """

        # LRP Composite-Regeln definieren
        composite_rules = {
            "low_level_hidden_layer_rule": "Epsilon",
            "mid_level_hidden_layer_rule": "Epsilon",
            "high_level_hidden_layer_rule": "Epsilon",
            "fully_connected_layers_rule": "Epsilon",
            "softmax_rule": "Epsilon",
        }

        composite = get_cnn_composite(model_name, composite_rules)
        component_attributor = ComponentAttribution(
            "Relevance", "CNN", layer_type
        )

        return composite, component_attributor

# Hilfsfunktion für cProfile
import cProfile

def profile_compute_lrp_pruning_mask(client, composite, component_attributor, pruning_rate, com_round):
    """
    Führt die compute_lrp_pruning_mask-Methode aus und profiliert sie mit cProfile.
    """
    if com_round == 1:
        print(f"Führe LRP-Pruning in Runde {com_round} durch...")

        # Profiling mit pyinstrument starten
        profiler = Profiler()
        profiler.start()

        # Profiling mit cProfile starten
        print("Starte cProfile für compute_lrp_pruning_mask...")
        pr = cProfile.Profile()
        pr.runctx(
            "client.compute_lrp_pruning_mask(composite=composite, component_attributor=component_attributor, pruning_rate=pruning_rate)",
            globals(),
            locals()
        )
        pr.dump_stats("output.prof")

        try:
            # Der relevante Codeblock
            global_concept_maps = client.compute_lrp_pruning_mask(
                composite=composite,
                component_attributor=component_attributor,
                pruning_rate=pruning_rate,
            )
            pruning_ops = GlobalPruningOperations(
                target_layer=torch.nn.Conv2d,
                layer_names=list(global_concept_maps.keys())
            )
            global_pruning_mask = pruning_ops.generate_global_pruning_mask(
                model=client.model,
                global_concept_maps=global_concept_maps,
                pruning_percentage=pruning_rate,
                least_relevant_first=True,
                device=client.device
            )

            print(f"LRP Pruning applied successfully in Round {com_round}.")
        except Exception as e:
            print(f"Failed to compute pruning mask: {e}")
            raise

        # Profiling mit pyinstrument stoppen und Ergebnisse speichern
        profiler.stop()

        # Profiling-Ergebnisse mit pyinstrument speichern
        with open("pruning_callgraph.txt", "w") as f:
            f.write(profiler.output_text(unicode=True, color=False))
        with open("pruning_callgraph.html", "w") as f:
            f.write(profiler.output_html())

        # Optional: Ergebnisse von cProfile in einen visuellen Callgraph konvertieren
        print("Erstelle Callgraph aus cProfile-Daten...")
        import os
        os.system("gprof2dot -f pstats output.prof | dot -Tpng -o callgraph.png")
