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

from collections import OrderedDict, defaultdict
import random
import yaml
from pxp import GlobalPruningOperations
from pxp import ComponentAttribution
from pxp.composites import *
from data.imagenet import ImageNetSubset, ImageNetSubset, get_sample_indices_for_class
from metrics.accuracy import compute_accuracy


'''data_dirs = {
        "images_lmdb": "/data/kaiclasen/BENv2.lmdb",
         "metadata_parquet": "/data/kaiclasen/metadata.parquet",
         "metadata_snow_cloud_parquet": "/data/kaiclasen/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    }'''

data_dirs = {
        "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
         "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
         "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    }

config_path = "CV4RS-orig/configs/test-config-resnet-p.yaml"

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

    '''def fed_avg(self, model_updates: list[dict]):
        assert len(model_updates) > 0, "Trying to aggregate empty update list"
        
        update_aggregation = {}
        for key in model_updates[0].keys():
            params = torch.stack([update[key] for update in model_updates], dim=0)
            avg = torch.mean(params, dim=0)
            update_aggregation[key] = avg
        
        return update_aggregation'''
    
    def fed_avg(self, model_updates: list[dict]):
        assert len(model_updates) > 0, "Trying to aggregate empty update list"
        
        update_aggregation = {}
        for key in model_updates[0].keys():
            # Check if the key exists in all updates
            if all(key in update for update in model_updates):
                params = torch.stack([update[key] for update in model_updates], dim=0)
                avg = torch.mean(params, dim=0)
                update_aggregation[key] = avg
            else:
                print(f"Skipping key {key} as it is missing in some updates.")
        
        return update_aggregation
    
    '''def fed_avg(self, model_updates: list[dict]):
        """
        Perform Federated Averaging of model updates.

        Args:
            model_updates (list[dict]): A list of state_dict updates from clients.

        Returns:
            dict: Aggregated model state_dict.
        """
        assert len(model_updates) > 0, "Trying to aggregate empty update list"
        
        update_aggregation = {}
        all_keys = set().union(*[update.keys() for update in model_updates])  # Alle Keys sammeln
        
        for key in all_keys:
            # Check if the key exists in at least one update
            available_updates = [update[key] for update in model_updates if key in update]
            
            if len(available_updates) > 0:
                # Average only for available updates
                params = torch.stack(available_updates, dim=0)
                avg = torch.mean(params, dim=0)
                update_aggregation[key] = avg
            else:
                print(f"[WARNING] Key {key} is missing in all updates. Skipping...")

        return update_aggregation'''



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
        pruner=None,
    ) -> None:
        global config_path
        with open(config_path, "r") as stream:
            self.configs = yaml.safe_load(stream)
        self.pruner = pruner
        self.pruning_mask = None
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
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

    def set_model(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model)
       
    # from here mask seeting and application
    def set_pruner(self, pruner):
        """
        Setzt den Pruner für den Client.

        Args:
            pruner (GlobalPruningOperations): Der Pruner.
        """
        self.pruner = pruner
 
    def set_pruning_mask(self, pruning_mask: OrderedDict):
        """
        Speichert die Pruning-Maske lokal im Client.

        Args:
            pruning_mask (OrderedDict): Die generierte Pruning-Maske.
        """
        self.pruning_mask = pruning_mask
        
        

    def train_one_round(self, epochs: int, validate: bool = False):
        """
        Trainiert das Modell für eine Runde mit oder ohne Pruning-Maske.

        Args:
            epochs (int): Anzahl der Trainings-Epochen.
            validate (bool): Führt eine Validierung nach dem Training durch, falls True.

        Returns:
            dict: Modell-Updates nach dem Training.
        """
        # [DEBUG] Speichere den aktuellen Zustand des Modells
        print("[DEBUG] Saving initial model state...")
        state_before = copy.deepcopy(self.model.state_dict())
        keys_before = set(state_before.keys())
        print(f"[DEBUG] Keys in state_before: {keys_before}")

        # Initialisiere Hook-Handles
        hook_handles = []
        
        # Prüfe, ob eine Pruning-Maske gesetzt ist
        if self.pruner is not None and self.pruning_mask is not None:
            print("[DEBUG] Pruning mask detected. Attempting to apply...")
            try:
                hook_handles = self.pruner.fit_pruning_mask(
                    self.model,
                    self.pruning_mask,
                ) or []  # Fallback auf eine leere Liste
            except Exception as e:
                print(f"[ERROR] Error applying pruning mask: {e}")
        
        # [DEBUG] Prüfen der Keys nach Pruning
        state_after_pruning = self.model.state_dict()
        keys_after_pruning = set(state_after_pruning.keys())
        missing_keys = keys_before - keys_after_pruning
        additional_keys = keys_after_pruning - keys_before
        print(f"[DEBUG] Keys missing after pruning: {missing_keys}")
        print(f"[DEBUG] Additional keys after pruning: {additional_keys}")

        # Initialisiere den Optimizer und die Loss-Funktion
        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)
        
        # Training über die angegebenen Epochen
        for epoch in range(1, epochs + 1):
            print(f"[INFO] Epoch {epoch}/{epochs}")
            print("-" * 10)
            self.train_epoch()
        
        # Validierung nach dem Training (optional)
        if validate:
            print("[INFO] Running validation...")
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)
        
        # Entferne die Hooks, falls sie gesetzt wurden
        for hook in hook_handles:
            hook.remove()
        
        # [DEBUG] Speichere den Zustand des Modells nach dem Training
        print("[DEBUG] Saving final model state...")
        state_after = self.model.state_dict()
        keys_after = set(state_after.keys())
        print(f"[DEBUG] Keys in state_after: {keys_after}")

        # [DEBUG] Prüfen der Änderungen an den Keys
        missing_keys_final = keys_before - keys_after
        additional_keys_final = keys_after - keys_before
        print(f"[DEBUG] Keys missing after training: {missing_keys_final}")
        print(f"[DEBUG] Additional keys after training: {additional_keys_final}")

        # Berechne die Differenz zwischen dem vorherigen und dem aktuellen Modellzustand
        model_update = {}
        for key, value_before in state_before.items():
            if key in state_after:  # Prüfen, ob der Key nach dem Training existiert
                value_after = state_after[key]
                diff = value_after.type(torch.DoubleTensor) - value_before.type(torch.DoubleTensor)
                model_update[key] = diff
            else:
                print(f"[WARNING] Key {key} is missing in state_after. Skipping...")

        # Rückgabe der Modell-Updates
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
            
            data = data.to(self.device)
            label_new=np.copy(labels)
           # label_new=self.change_sizes(label_new)
            label_new = torch.from_numpy(label_new).to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(data)
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
        global config_path
        with open(config_path, "r") as stream:
            self.configs = yaml.safe_load(stream)
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
        
        
        self.pruning_patches = []
        self.dataset = BENv2DataSet(
            data_dirs=data_dirs,
            split="train",  # Für Trainingsdaten
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=["Finland", "Ireland", "Serbia"], seasons="Summer"),
        )
        self.pruning_dataset = PruneDataSet(
            pruning_patches=self.pruning_patches,  # Die gesammelten Patches
            data_dirs=data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
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
                
    def create_pruning_loader(self, pruning_patches: list[str]) -> DataLoader:
        """
        Create a DataLoader for the given pruning patches.

        Args:
            pruning_patches (list[str]): List of patch IDs to include in the pruning dataset.

        Returns:
            DataLoader: DataLoader for the pruning patches.
        """
        # Standardwerte für fehlende Konfigurationskeys
        batch_size = self.configs.get("pruning_dataloader_batchsize", 32)  # Standardwert: 32
        num_workers = self.configs.get("num_workers", 0)  # Standardwert: 0

        # Erstelle das Pruning-Dataset und speichere es als Attribut
        self.pruning_dataset = PruneDataSet(
            pruning_patches=pruning_patches,  # Die gesammelten Patches
            data_dirs=data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
        )

        # Erstelle den Pruning-Loader
        prune_loader = DataLoader(
            self.pruning_dataset,
            batch_size=min(256, len(self.pruning_dataset)),  # Dynamische Anpassung der Batch-Größe
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

        return prune_loader


    def train(self, communication_rounds: int, epochs: int):
        start = time.perf_counter()
        for com_round in range(1, communication_rounds + 1):
            print("=" * 50)
            print("ROUND {}/{}".format(com_round, communication_rounds))
            print("=" * 50)
        

            # Pruning mask generation
            if com_round == 2:
                # Trainings- und Validierungsdatensatz setzen
                train_set = self.dataset
                #val_set = self.validation_set
                
                
                
                # Liste aller verfügbaren Klassen
                all_classes = [
                    'Transitional woodland, shrub', 'Coastal wetlands', 'Urban fabric', 'Arable land',
                    'Moors, heathland and sclerophyllous vegetation', 'Inland wetlands', 'Permanent crops',
                    'Industrial or commercial units', 'Mixed forest', 'Broad-leaved forest',
                    'Natural grassland and sparsely vegetated areas',
                    'Land principally occupied by agriculture, with significant areas of natural vegetation',
                    'Marine waters', 'Beaches, dunes, sands', 'Coniferous forest', 'Pastures',
                    'Inland waters', 'Agro-forestry areas', 'Complex cultivation patterns'
                ]

                '''from collections import defaultdict

                # Initialisiere ein Dictionary, um die Klassenhäufigkeiten in den Patches zu überwachen
                class_counts = defaultdict(int)
                collected_classes = set()

                # Iteriere über die Patches und sammle Klassen, bis die gewünschte Anzahl erreicht ist
                for patch in train_set.patches:
                    patch_labels = train_set.BENv2Loader.lbls[patch]

                    # Prüfe, ob der Patch neue Klassen enthält
                    new_labels = [label for label in patch_labels if label not in collected_classes]

                    # Füge den Patch hinzu, wenn er neue Klassen enthält
                    if new_labels:
                        self.pruning_patches.append(patch)

                        # Aktualisiere die gesammelten Klassen und deren Häufigkeiten
                        for label in patch_labels:
                            collected_classes.add(label)
                            class_counts[label] += 1

                    # Prüfe, ob die gewünschte Anzahl eindeutiger Klassen erreicht wurde
                    if len(collected_classes) >= self.configs["domain_restriction_classes"]:
                        break  # Abbruch, wenn die gewünschte Anzahl an Klassen gesammelt wurde

                # Zähle die Häufigkeit jeder Klasse in den gesammelten Patches
                total_class_counts = defaultdict(int)
                for patch in self.pruning_patches:
                    patch_labels = train_set.BENv2Loader.lbls[patch]
                    for label in patch_labels:
                        total_class_counts[label] += 1

                # Ausgabe der Ergebnisse
                print(f"Finale Anzahl der Pruning-Patches: {len(self.pruning_patches)}")
                print(f"Anzahl eindeutiger Klassen: {len(collected_classes)}")
                print(f"Klassenverteilung in den Pruning-Patches (Häufigkeiten): {dict(total_class_counts)}")



                #self.prune_loader = self.create_pruning_loader(pruning_patches)
                
                self.prune_loader = create_prune_loader(self.clients[0].train_loader, self.pruning_patches)
                '''
                
                # 30 patch big subset of trainloader
                original_dataset = self.clients[0].train_loader.dataset
                assert len(original_dataset) >= 30, "Das Dataset enthält weniger als 30 Patches!"
                selected_indices = list(range(30))  # Nimm die ersten 30 Patches
                subset_dataset = torch.utils.data.Subset(original_dataset, selected_indices)

                # Neuen Dataloader mit dem Subset erstellen
                train_loader1 = torch.utils.data.DataLoader(
                    subset_dataset,
                    batch_size=self.clients[0].train_loader.batch_size,
                    shuffle=False,
                    num_workers=self.clients[0].train_loader.num_workers,
                    pin_memory=self.clients[0].train_loader.pin_memory
                )
                print(f"Created train_loader1 with {len(subset_dataset)} patches.")

                
                suggested_composite = {
                    "low_level_hidden_layer_rule": self.configs["low_level_hidden_layer_rule"],
                    "mid_level_hidden_layer_rule": self.configs["mid_level_hidden_layer_rule"],
                    "high_level_hidden_layer_rule": self.configs["high_level_hidden_layer_rule"],
                    "fully_connected_layers_rule": self.configs["fully_connected_layers_rule"],
                    "softmax_rule": self.configs["softmax_rule"],
                }
                
                if self.configs["model_architecture"] == "vit_b_16":
                    composite = get_vit_composite(
                        self.configs["model_architecture"], suggested_composite
                    )
                else:
                    composite = get_cnn_composite(
                        self.configs["model_architecture"], suggested_composite
                    )
                    
                    
                layer_types = {
                    "Softmax": torch.nn.Softmax,
                    "Linear": torch.nn.Linear,
                    "Conv2d": torch.nn.Conv2d,
                }
                            
                print("Starting relevance computation and pruning mask generation.")
                # Laden der relevanten Konfigurationen
                pruning_rates = self.configs["pruning_rates"]

                # Initialisierung des Relevance-Attributors
                component_attributor = ComponentAttribution(
                    "Relevance",
                    "CNN",  # Annahme: ResNet wird verwendet
                    layer_types[self.configs["pruning_layer_type"]],
                )
                
                # Berechnung der Relevanzen
                #print(f"Calling attribute with prune_loader: {self.prune_loader}, composite: {composite}")
                #print(f"Calling attribute with prune_loader: {train_loader1}, composite: {composite}")
                #print(f"model: {self.model}, device: {self.device}")
                try:
                    components_relevances = component_attributor.attribute(
                        self.model,
                        train_loader1, # self.prune_loader,
                        composite,
                        abs_flag=True,
                        device=self.device,
                    )
                except Exception as e:
                    print(f"Exception occurred in attribute: {e}")
                # Debugging: Überprüfen des Rückgabetyps und des Inhalts
                print(f"Type of components_relevances: {type(components_relevances)}")
                # Iteriere über das OrderedDict und gib die Relevanzen aus
                '''for layer_name, relevance in components_relevances.items():
                    print(f"Layer: {layer_name}")
                    print(f"Relevance shape: {relevance.shape}")
                    print(f"Relevance values: {relevance}")     
                    total_relevance = relevance.sum().item()
                    print(f"Total layer relevance: {total_relevance}")               
                    print("-" * 50)'''
                    
                for layer_name, relevance in components_relevances.items():
                    total_relevance = relevance.sum().item()
                    print(f"Layer: {layer_name}")
                    print(f"Total layer relevance: {total_relevance}")
                    
                    print("-" * 50)
                    
                # neu
                '''acc_top1 = compute_accuracy(
                    self.model,
                    custom_validation_dataloader,
                    self.device,
                )
                
                print(f"Initial accuracy: top1={acc_top1}")'''
                """
                Experiment's main loop
                """
                layer_names = component_attributor.layer_names
                pruner = GlobalPruningOperations(
                    layer_types[self.configs["pruning_layer_type"]],
                    layer_names,
                )
                #top1_acc_list = []
                #progress_bar = tqdm.tqdm(total=len(pruning_rates))
                
                global_pruning_mask = OrderedDict([])
                
                # prune the model based on the
                # pre-computed attibution flow
                # (relevance values)
                try:
                    global_pruning_mask = pruner.generate_global_pruning_mask(
                        self.model,
                        components_relevances,
                        pruning_percentage=0.1,
                        subsequent_layer_pruning=self.configs["subsequent_layer_pruning"],
                        least_relevant_first=self.configs["least_relevant_first"],
                        device=self.device,
                    )
                except Exception as e:
                    print(f"Error during pruning mask generation: {e}")
            
                print(f"Global Pruning Mask: {global_pruning_mask}")
                
                # Debugging-Code einfügen, um Schlüssel zu vergleichen
                keys_before = set(self.model.state_dict().keys())
                keys_after = set(global_pruning_mask.keys()) if global_pruning_mask else set()

                # Unterschied prüfen
                missing_keys = keys_before - keys_after
                additional_keys = keys_after - keys_before

                print(f"Keys missing after pruning: {missing_keys}")
                print(f"Additional keys after pruning: {additional_keys}")

                print("Distributing pruning mask to clients...")
                for client in self.clients:
                    client.set_pruner(pruner)
                    client.set_pruning_mask(global_pruning_mask)
        
                # Our pruning gets applied by masking the
                # activation of layers via forward hooks.
                # Therefore hooks are returned for later
                # removal

                # empty up the GPU memory and CUDA cache, model and dataset
                #progress_bar.close()
                #del pruner
                #torch.cuda.empty_cache()

                '''top1_auc = compute_auc(top1_acc_list, pruning_rates)
                if self.configs["wandb"]:
                    wandb.log({"top1_auc": top1_auc})
                    print(f"Logged the AUC of the Top1 Accuracy to wandb!")

                print(f"Top1 AUC: {top1_auc}")'''
                

            self.communication_round(epochs)
            report = self.validation_round()

            self.results = update_results(self.results, report, self.num_classes)
            print_micro_macro(report)

            for client in self.clients:
                client.set_model(self.model)
                
        self.train_time = time.perf_counter() - start

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
                label_new=np.copy(labels)
               # label_new=self.change_sizes(label_new)

                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()

                predicted_probs += list(probs)

                y_true += list(label_new)

        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)

        y_true = np.asarray(y_true)
        report = get_classification_report(
            y_true, y_predicted, predicted_probs, self.dataset_filter
        )
        return report

    def communication_round(self, epochs: int):
        # here the clients train
        # TODO: could be parallelized
        model_updates = [client.train_one_round(epochs) for client in self.clients]

        # parameter aggregation
        update_aggregation = self.aggregator.fed_avg(model_updates)

        # update the global model. original code
        '''global_state_dict = self.model.state_dict()
        for key, value in global_state_dict.items():
            update = update_aggregation[key].to(self.device)
            global_state_dict[key] = value + update
        self.model.load_state_dict(global_state_dict)'''
        
        # Update the global model
        global_state_dict = self.model.state_dict()

        for key, value in global_state_dict.items():
            # Skip missing keys (e.g., 'conv1.bias' and 'encoder.1.bias') for now
            if key not in update_aggregation:
                print(f"Skipping key: {key}, not found in update aggregation")
                continue
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
        
        
from utils.BENv2_dataset import BENv2DataSet

class PruneDataSet(BENv2DataSet):
    def __init__(self, pruning_patches, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Setze die Patches auf die Pruning-Patches
        self.patches = pruning_patches
        # Sortiere für Konsistenz
        self.patches.sort()

def validate_prune_loader(train_loader, prune_loader, pruning_dataset):
    print("[INFO] Validierung des Prune Loaders...")

    for batch_idx, (train_batch, prune_batch) in enumerate(zip(train_loader, prune_loader)):
        for i, (train_item, prune_item) in enumerate(zip(train_batch, prune_batch)):
            # Prüfe nur die Shape-Dimensionen, nicht die exakte Anzahl
            assert train_item.shape[1:] == prune_item.shape[1:], \
                f"Shape mismatch in batch {batch_idx}, element {i}: {train_item.shape} != {prune_item.shape}"

    print("[SUCCESS] Prune Loader Struktur validiert!")


'''def validate_prune_loader(train_loader, prune_loader, pruning_dataset):
    """
    Validiert die Struktur des prune_loader im Vergleich zum train_loader
    und prüft, ob die Daten korrekt geladen werden.

    Args:
        train_loader (DataLoader): Originaler Trainings-Loader.
        prune_loader (DataLoader): Pruning-Loader.
        pruning_dataset (PruneDataSet): Pruning-Dataset.
    """
    print("[INFO] Validierung des Prune Loaders...")

    # Prüfen, ob beide Loader gleichartige Datenstrukturen zurückgeben
    for batch_idx, (train_batch, prune_batch) in enumerate(zip(train_loader, prune_loader)):
        # Anzahl der Elemente im Batch prüfen
        assert len(train_batch) == len(prune_batch), f"Batch structure mismatch: train_batch has {len(train_batch)} elements, prune_batch has {len(prune_batch)} elements."

        # Iteriere über die Elemente in einem Batch
        for i, (train_item, prune_item) in enumerate(zip(train_batch, prune_batch)):
            # Typen vergleichen
            assert type(train_item) == type(prune_item), f"Type mismatch in batch {batch_idx}, element {i}: {type(train_item)} != {type(prune_item)}"
            
            # Shapes vergleichen (nur für Tensoren relevant)
            if isinstance(train_item, torch.Tensor):
                assert train_item.shape == prune_item.shape, f"Shape mismatch in batch {batch_idx}, element {i}: {train_item.shape} != {prune_item.shape}"

    print("[SUCCESS] Prune Loader Struktur validiert!")

    # Debugging innerhalb des Pruning Datasets
    print("[INFO] Debugging des Prune Datasets...")
    for patch in pruning_dataset.patches:
        assert patch in pruning_dataset.BENv2Loader.lbls, f"Patch {patch} nicht in BENv2Loader vorhanden"
    print("[SUCCESS] Alle Pruning-Patches haben gültige Labels!")

    # Debugging der geladenen Daten
    print("[INFO] Debugging der geladenen Daten...")
    for idx, img, _, key, labels in prune_loader:
        print(f"[DEBUG] Index: {idx}, Image Shape: {img.shape}, Key: {key}, Labels: {labels}")
        break  # Nur eine Batch prüfen'''


from torch.utils.data import DataLoader, Subset

def create_prune_loader(train_loader, pruning_patches):
    """
    Erstellt einen Prune-Loader basierend auf dem train_loader, indem nur die Patches
    in `pruning_patches` beibehalten werden.

    Args:
        train_loader (DataLoader): Der ursprüngliche Trainings-Loader.
        pruning_patches (list): Liste der Patches, die im Prune-Loader enthalten sein sollen.

    Returns:
        DataLoader: Der Prune-Loader mit derselben Struktur wie der train_loader.
    """
    print("[INFO] Erstelle Prune Loader...")

    # Kopiere das Dataset aus dem train_loader
    train_dataset = train_loader.dataset  # Das originale BENv2DataSet
    prune_dataset = copy.deepcopy(train_dataset)

    # Filtere die Patches, die nicht in pruning_patches sind
    prune_dataset.patches = [patch for patch in prune_dataset.patches if patch in pruning_patches]
    print(f"[INFO] {len(prune_dataset.patches)} Patches nach Filterung übrig.")

    # Aktualisiere abhängige Attribute
    prune_dataset.BENv2Loader.lbls = {patch: lbl for patch, lbl in prune_dataset.BENv2Loader.lbls.items() if patch in pruning_patches}
    prune_dataset.BENv2Loader.lbl_key_set = set(prune_dataset.patches)

    # Erstelle einen neuen DataLoader basierend auf dem gefilterten Dataset
    prune_loader = torch.utils.data.DataLoader(
        dataset=prune_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,  # Da Reihenfolge wichtig sein könnte
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last,
    )

    print("[SUCCESS] Prune Loader erfolgreich erstellt.")
    return prune_loader
