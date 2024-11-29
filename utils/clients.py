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
from pxp.prune import GlobalPruningOperations
from utils.pruning_utils import apply_pruning, apply_lrp_pruning
import pxp.prune
print(pxp.prune.__file__)


from pyinstrument import Profiler
from utils.profiling_utils import profile_with_pyinstrument, profile_with_cprofile
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
        optimizer_kwargs: dict = {"lr": 0.0002, "weight_decay": 0},    # 0.001
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {
            "reduction": "mean",
            "pos_weight": torch.tensor([2.0]).cuda(),   # Regulation
        },
        num_classes: int = 19,
        device: torch.device = torch.device("cpu"),
        dataset_filter: str = "serbia",
        data_dirs: dict = None,
    ) -> None:
        self.model = model
        self._initialize_weights()  # Initialize weights Paul
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.pruning_mask = None  # Initialize pruning mask Paul
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

    def _initialize_weights(self):
        """
        Initializes weights in the model using Xavier or Kaiming initialization.
        """
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() > 1:  # Check if the parameter is a weight matrix
                torch.nn.init.xavier_uniform_(param)  # Alternatively: kaiming_uniform_
                #print(f"Initialized {name} with Xavier uniform.")
            elif "bias" in name:
                torch.nn.init.zeros_(param)
                #print(f"Initialized {name} with zeros.")

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

        # Initialize optimizer and scheduler
        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            print("-" * 10)

            # Training for one epoch
            self.train_epoch()

            # Step the scheduler
            self.scheduler.step()
            print(f"Updated learning rate: {self.scheduler.get_last_lr()}")

        if validate:
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)

        state_after = self.model.state_dict()

        # Compute model updates
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
        epsilon = 1e-8  # Small value to avoid numerical instability

        for idx, batch in enumerate(tqdm(self.train_loader, desc="training")):
            data = batch[1].cuda()
            labels = torch.from_numpy(np.copy(batch[4])).cuda()

            # Debugging: Check input and label tensors
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"NaN or Inf detected in input data for batch {idx}")
                raise ValueError("Invalid input data detected.")
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print(f"NaN or Inf detected in labels for batch {idx}")
                raise ValueError("Invalid labels detected.")

            self.optimizer.zero_grad()

            logits = self.model(data)

            # Add epsilon to logits for numerical stability. not the solution to NaN. so
            # taking it out again, to not influence performance
            #logits = logits + epsilon

            # Debugging: Check logits after adjustment
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"NaN or Inf detected in logits after adding epsilon for batch {idx}")
                raise ValueError("Invalid logits detected.")

            loss = self.criterion(logits, labels)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Debugging: Check gradients for NaN or Inf
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"NaN or Inf detected in gradient for parameter {name}")
                        raise ValueError("Invalid gradient detected.")

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
        Berechnet die LRP-Pruning-Maske basierend auf globalen Relevanzwerten.

        Args:
            composite: Das Composite-Objekt für LRP.
            component_attributor: Das Component Attribution-Objekt.
            pruning_rate (float): Der Anteil der zu prunenden Parameter.

        Returns:
            dict: Die generierte globale Pruning-Maske.
        """
        print(f"[DEBUG] Starte Berechnung der LRP-Pruning-Maske für Land: Finland")
        print(f"[DEBUG] Pruning-Rate: {pruning_rate}")

        # Lade Dataloader
        dataloader = self.get_country_dataloader("Finland", batch_size=16, num_workers=4)
        print(f"[DEBUG] Dataloader für 'Finland' geladen mit Batch-Größe 16 und 4 Arbeitern.")

        # Berechnung der Relevanzwerte
        print("[DEBUG] Starte Berechnung der Relevanzwerte...")
        global_concept_maps = component_attributor.attribute(
            model=self.model,
            dataloader=(
                (batch[0].float(), batch[-1])  # Passe die Struktur der Batch an
                for batch in dataloader
            ),
            attribution_composite=composite,
            abs_flag=True,
            device=self.device,
        )
        print("[DEBUG] Relevanzwerte berechnet. Anzahl der Layer mit Relevanzwerten:", len(global_concept_maps))

        # Debugging: Überprüfe Struktur und Inhalt von global_concept_maps
        print("[DEBUG] Überprüfe Struktur und Inhalt von global_concept_maps vor Pruning-Maske...")
        for layer_name, value in global_concept_maps.items():
            if value is None:
                print(f"[ERROR] Layer '{layer_name}' hat None-Wert in global_concept_maps.")
            elif isinstance(value, dict):
                print(f"[DEBUG] Layer '{layer_name}' enthält Dictionary: Keys = {list(value.keys())}")
            elif isinstance(value, torch.Tensor):
                print(f"[DEBUG] Layer '{layer_name}' enthält Tensor: Shape = {value.shape}")
            else:
                print(f"[WARNING] Layer '{layer_name}' enthält unbekannten Typ: {type(value)}")

        # Validierung der global_concept_maps
        for layer_name, value in global_concept_maps.items():
            if value is None:
                raise ValueError(
                    f"[ERROR] Layer '{layer_name}' hat None-Wert in global_concept_maps. Überprüfe die Attributberechnung.")
            if isinstance(value, dict):
                if "relevance" not in value:
                    raise KeyError(
                        f"[ERROR] Layer '{layer_name}' enthält Dictionary ohne Key 'relevance'. Inhalt: {list(value.keys())}")
            elif not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"[ERROR] Layer '{layer_name}' enthält ungültigen Typ: {type(value)}. Erwartet: torch.Tensor oder dict.")

        # Initialisiere GlobalPruningOperations
        print("[DEBUG] Initialisiere GlobalPruningOperations...")
        target_layer = torch.nn.Conv2d
        layer_names = [
            name for name, _ in self.model.named_modules() if isinstance(_, target_layer)
        ]

        pruning_operations = GlobalPruningOperations(
            target_layer=target_layer,
            layer_names=layer_names,
        )
        print(f"[DEBUG] Ziel-Layer: {pruning_operations.layer_names}")

        # Generiere globale Pruning-Maske
        print("[DEBUG] Starte Generierung der globalen Pruning-Maske...")
        global_pruning_mask = pruning_operations.generate_global_pruning_mask(
            model=self.model,
            global_concept_maps=global_concept_maps,
            pruning_percentage=pruning_rate,
            subsequent_layer_pruning="Both",  # Passe dies je nach Bedarf an
            least_relevant_first=True,
            device=self.device,
        )
        print(f"[DEBUG] Globale Pruning-Maske generiert für {len(global_pruning_mask)} Layer")

        # Debug-Print für die globale Pruning-Maske
        for layer_name, mask in global_pruning_mask.items():
            print(f"[DEBUG] Layer: {layer_name} | Masken-Typen: {list(mask.keys())}")
            if "weight" in mask:
                print(
                    f"  Gewicht-Maske - Shape: {mask['weight'].shape}, Min: {mask['weight'].min()}, Max: {mask['weight'].max()}")

        # Validierung: Existieren alle Layer in global_pruning_mask im Modell?
        missing_layers = [
            layer_name for layer_name in global_pruning_mask.keys()
            if ModelLayerUtils.get_module_from_name(self.model, layer_name) is None
        ]
        if missing_layers:
            raise ValueError(f"[ERROR] Die folgenden Layer existieren nicht im Modell: {missing_layers}")

        # Registriere Forward Hooks
        print("[DEBUG] Registriere Forward Hooks für Pruning...")
        if global_pruning_mask:
            self.pruning_hook_handles = pruning_operations.fit_pruning_mask(self.model, global_pruning_mask)
            print(
                f"[DEBUG] Forward Hooks registriert: {len(self.pruning_hook_handles) if self.pruning_hook_handles else 'Keine Hooks'}")
        else:
            print("[ERROR] Keine gültigen Pruning-Masken verfügbar.")
            raise ValueError("Pruning-Masken konnten nicht generiert werden.")

        # Debugging: Überprüfe Zustand von global_concept_maps nach Hook-Registrierung
        print("[DEBUG] Überprüfe Struktur von global_concept_maps nach Hook-Registrierung...")
        for layer_name, value in global_concept_maps.items():
            if value is None:
                print(f"[ERROR] Layer '{layer_name}' hat None-Wert.")
            elif isinstance(value, torch.Tensor):
                print(f"[DEBUG] Layer '{layer_name}' hat Tensor mit Shape {value.shape}.")
            else:
                print(f"[WARNING] Layer '{layer_name}' hat unbekannten Typ: {type(value)}.")

        # Validierung: Funktioniert 'conv1' korrekt nach der Pruning-Maske?
        conv1_layer = ModelLayerUtils.get_module_from_name(self.model, "conv1")
        if conv1_layer is None:
            print("[ERROR] Layer 'conv1' konnte nach der Pruning-Maske nicht gefunden werden.")
        else:
            print(f"[DEBUG] Layer 'conv1' nach Pruning-Maske erfolgreich gefunden: {conv1_layer}.")

        return global_pruning_mask

    def remove_pruning_hooks(self):
        """
        Entfernt alle registrierten Forward Hooks.
        """
        if hasattr(self, "pruning_hook_handles") and self.pruning_hook_handles is not None:
            for hook in self.pruning_hook_handles:
                hook.remove()
            self.pruning_hook_handles = []
            print("All forward hooks removed.")
        else:
            print("[WARNING] No pruning hooks to remove.")

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
        pruning_rate = 0.3  # Pruning rate
        global_pruning_mask = None  # Initialize pruning mask

        # Initialize LRP Pruning
        print("Initializing LRP Pruning...")
        composite, component_attributor = self.initialize_lrp_pruning("resnet50", torch.nn.Conv2d)
        print("LRP initialized successfully.")

        for com_round in range(1, communication_rounds + 1):
            print(f"=== Round {com_round}/{communication_rounds} ===")

            # Apply pruning mask (if available)
            if global_pruning_mask is not None:
                print(f"Applying pruning mask for round {com_round}...")
                self.pruning_hook_handles = GlobalPruningOperations(
                    torch.nn.Conv2d, list(global_pruning_mask.keys())
                ).fit_pruning_mask(self.model, global_pruning_mask)

            try:
                # Training and communication
                print(f"Training and communication for Round {com_round}...")
                self.communication_round(epochs)

                # Validation and error handling
                print(f"Starting validation after Round {com_round}...")
                report = self.validation_round()
                self.results = update_results(self.results, report, self.num_classes)
                print_micro_macro(report)

            except Exception as e:
                print(f"Error during training or validation: {e}")
                raise

            # LRP-Pruning nach der ersten Kommunikationsrunde
            if com_round == 1:  # Beispiel: Nur nach der ersten Runde prunen
                print(f"Führe LRP-Pruning in Runde {com_round} durch...")

                # Profiling starten
                profiler = Profiler()
                profiler.start()

                try:
                    # Der relevante Codeblock
                    print(f"Führe LRP-Pruning in Runde {com_round} durch...")
                    global_concept_maps = self.compute_lrp_pruning_mask(
                        composite=composite,
                        component_attributor=component_attributor,
                        pruning_rate=pruning_rate,
                    )
                    pruning_ops = GlobalPruningOperations(
                        target_layer=torch.nn.Conv2d,
                        layer_names=list(global_concept_maps.keys())
                    )
                    global_pruning_mask = pruning_ops.generate_global_pruning_mask(
                        model=self.model,
                        global_concept_maps=global_concept_maps,
                        pruning_percentage=pruning_rate,
                        subsequent_layer_pruning=True,  # Propagiere Pruning
                        least_relevant_first=True,
                        device=self.device,
                    )

                    print(f"LRP Pruning applied successfully in Round {com_round}.")
                except Exception as e:
                    print(f"Failed to compute pruning mask: {e}")
                    raise

                # Profiling stoppen und Ergebnis anzeigen
                profiler.stop()
                with open("pruning_callgraph.txt", "w") as f:
                    f.write(profiler.output_text(unicode=True, color=False))
                #with open("pruning_callgraph.html", "w") as f:
                #    f.write(profiler.output_html())

            # Remove hooks at the end of the round
            self.remove_pruning_hooks()

        # Finalize training time
        self.train_time = time.perf_counter() - start

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
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="test")):
                data = batch[1].to(self.device)
                labels = batch[4]

                # Validate input data
                if torch.isnan(data).any() or torch.isinf(data).any():
                    print(f"NaN or Inf detected in input data in batch {batch_idx}")
                    raise ValueError("Invalid input data detected.")

                # Validate labels
                if torch.isnan(labels).any() or torch.isinf(labels).any():
                    print(f"NaN or Inf detected in labels in batch {batch_idx}")
                    raise ValueError("Invalid labels detected.")

                logits = self.model(data)

                # Validate logits
                if torch.isnan(logits).any() or torch.isinf(logits).any(): # Problemo
                    print(f"NaN or Inf detected in logits in batch {batch_idx}")
                    print(f"Logits stats - max: {logits.max()}, min: {logits.min()}, mean: {logits.mean()}")
                    raise ValueError("Invalid logits detected.")

                probs = torch.sigmoid(logits)

                # Validate probabilities
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print(f"NaN or Inf detected in probabilities in batch {batch_idx}")
                    print(f"Probabilities stats - max: {probs.max()}, min: {probs.min()}, mean: {probs.mean()}")
                    raise ValueError("Invalid probabilities detected.")

                predicted_probs += list(probs.cpu().numpy())  # Keep probabilities for metrics
                y_true += list(labels.numpy())

        predicted_probs = np.asarray(predicted_probs)
        y_true = np.asarray(y_true)

        # Validate final arrays
        if np.isnan(predicted_probs).any():
            print("NaN detected in predicted probabilities array.")
            raise ValueError("Predicted probabilities contain NaN values.")
        if np.isnan(y_true).any():
            print("NaN detected in true labels array.")
            raise ValueError("True labels contain NaN values.")

        report = get_classification_report(
            y_true, predicted_probs >= 0.5, predicted_probs, self.dataset_filter
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

class ModelLayerUtils:
    @staticmethod
    def get_module_from_name(model, layer_name):
        """
        Retrieves a layer/module from a model by its name.

        Args:
            model (torch.nn.Module): The model containing the layers.
            layer_name (str): The name of the layer to retrieve.

        Returns:
            torch.nn.Module or None: The layer/module if found, else None.
        """
        try:
            module = dict(model.named_modules()).get(layer_name, None)
            return module
        except Exception as e:
            print(f"[ERROR] Fehler beim Abrufen des Layers '{layer_name}': {e}")
            return None
