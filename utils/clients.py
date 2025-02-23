import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
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
from torch.profiler import profile, record_function, ProfilerActivity


data_dirs = {
        "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
         "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
         "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    }

config_path = "../CV4RS-orig/configs/test-config-resnet-p.yaml"

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
        scenario: int,
        scenario1_split: pd.DataFrame,
        batch_size: int = 256,
        num_workers: int = 2,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 19,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        dataset_filter: str = "serbia",
    ) -> None:
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.results = init_results(self.num_classes)
        self.pruner = None
        self.pruning_mask = None
        
        self.dataset = BENv2DataSet(
        data_dirs=data_dirs,
        # For Mars use these paths
        split="train",
        img_size=(10, 120, 120),
        include_snowy=False,
        include_cloudy=False,
        patch_prefilter=PreFilter(scenario1_split if scenario==1 else pd.read_parquet(data_dirs["metadata_parquet"]), countries=csv_path, #TODO ME was before [csv_path], # to enable passing list of csv_paths
                                  seasons=["Summer"]),
        normalize=True  # standardisation
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
        patch_prefilter=PreFilter(scenario1_split if scenario==1 else pd.read_parquet(data_dirs["metadata_parquet"]), countries=csv_path, #TODO ME was before [csv_path], # to enable passing list of csv_paths
                                  seasons=["Summer"]),
        normalize=True  # standardisation
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
        
    def set_pruner_and_mask(self, pruner: GlobalPruningOperations, pruning_mask: OrderedDict):
        """
        Speichert den Pruner und die Pruning-Maske für diesen Client.
        Args:
            pruner (GlobalPruningOperations): Der vom GlobalClient generierte Pruner.
            pruning_mask (OrderedDict): Die vom GlobalClient generierte Pruning-Maske.
        """
        self.pruner = pruner
        self.pruning_mask = pruning_mask
        print("[INFO] Pruner and pruning mask received and stored.")

    def train_one_round(self, epochs: int, validate: bool = False):
        state_before = copy.deepcopy(self.model.state_dict())

        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)

            self.train_epoch()
        
        if validate:
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)
            
        # prune again here: if mask is not none, then prune
        if self.pruning_mask is not None:
                #print("-" * 30)
                print("Prune before sending to global...")
                #print("-" * 30)
                self.pruner.fit_pruning_mask(
                    self.model,
                    self.pruning_mask,
                )
                
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
            
            data = batch[1].to(self.device)
            labels = batch[4].to(self.device)
            label_new = labels.clone().to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(data)
            loss = self.criterion(logits, label_new)
            loss.backward()
            
            del logits, data, labels
            torch.cuda.empty_cache()
            
            if self.pruning_mask is not None:
                #print("-" * 30)
                #print("Applying pruning mask...")
                #print("-" * 30)
                hook_handles = self.pruner.fit_pruning_mask(
                    self.model,
                    self.pruning_mask,
                )

            self.optimizer.step()
    
    def get_validation_results(self):
        return self.results

class GlobalClient:
    def __init__(
        self,
        model: torch.nn.Module,
        scenario: int,
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
        self.layer_types = {
            key: getattr(torch.nn, value) for key, value in self.configs["layer_types"].items()
        }
        self.scenario = scenario
        self.pruning_round = self.configs.get("pruning_round", 4)
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
        self.results = init_results(self.num_classes)
        shuffled_metadata = pd.read_parquet(data_dirs["metadata_parquet"]).sample(frac=1)
        df_splits = np.array_split(shuffled_metadata, len(csv_paths))

        self.clients = [
                        FLCLient(copy.deepcopy(self.model), lmdb_path, val_path,csv_path=(csv_paths if 1==scenario else csv_path), #TODO ME csv_pathS  ---- THIS DECIDES WHETHER ONE COUNTRY PER CLIENT OR MULTIPLE
                            scenario=scenario, scenario1_split=scenario1_split, # introduced this for scenatio1
                            num_classes=num_classes, batch_size=256, dataset_filter=dataset_filter, device=self.device) #TODO ME SET BATCH SIZE TO 512
            for csv_path,scenario1_split in zip(csv_paths,df_splits)
        ]
        print("\ninit GLOBALClient VALIDATION dataset and dataloader")
        
        
        self.validation_set = BENv2DataSet( 
        data_dirs=data_dirs,
        split="test",
        img_size=(10, 120, 120),
        include_snowy=False,
        include_cloudy=False,
        patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=["Finland","Ireland","Serbia"], seasons="Summer"),  #"Finland", "Ireland", "Serbia", "Austria", "Belgium", "Lithuania", "Portugal", "Switzerland"
        normalize=True # standardisation
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
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=["Finland","Ireland","Serbia"], seasons="Summer"),  #"Finland", "Ireland", "Serbia"
            normalize=True # standardisation
        )
        
        self.pruning_dataset = PruneDataSet(
            pruning_patches=self.pruning_patches,
            data_dirs=data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            normalize=True  # standardisation
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
        batch_size = self.configs.get("pruning_dataloader_batchsize", 32)  # Standardwert: 32
        num_workers = self.configs.get("num_workers", 1)  # Standardwert: 1

        self.pruning_dataset = PruneDataSet(
            pruning_patches=pruning_patches,  
            data_dirs=data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
        )

        prune_loader = DataLoader(
            self.pruning_dataset,
            batch_size=min(256, len(self.pruning_dataset)), 
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
            if com_round == self.pruning_round:
                self.prepare_pruning_loader()
                global_pruning_mask = self.compute_relevance_and_generate_mask()

                # Print statistics and distribute mask
                self.print_pruning_statistics(global_pruning_mask)
                self.distribute_pruning_mask(global_pruning_mask)

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


    def prepare_pruning_loader(self):
        """
        Prepares the pruning dataset and DataLoader.
        """
        train_set = self.dataset
        class_counts = defaultdict(int)
        collected_classes = set()

        for patch in train_set.patches:
            patch_labels = train_set.BENv2Loader.lbls[patch]
            new_labels = [label for label in patch_labels if label not in collected_classes]

            if new_labels:
                self.pruning_patches.append(patch)
                for label in patch_labels:
                    collected_classes.add(label)
                    class_counts[label] += 1

            if len(collected_classes) >= self.configs["domain_restriction_classes"]:
                break  

        total_class_counts = defaultdict(int)
        for patch in self.pruning_patches:
            patch_labels = train_set.BENv2Loader.lbls[patch]
            for label in patch_labels:
                total_class_counts[label] += 1

        print(f"Final number of pruning patches: {len(self.pruning_patches)}")
        print(f"Number of unique classes: {len(collected_classes)}")
        print(f"Class distribution in the pruning patches (frequencies): {dict(total_class_counts)}")

        self.prune_loader = create_prune_loader(self.clients[0].train_loader, self.pruning_patches)

        '''# RANDOM FEW-SHOT SAMPLE SUBSET (30 patches)
        original_dataset = self.clients[0].train_loader.dataset
        assert len(original_dataset) >= 30, "Dataset contains less then 30 patches!"
        selected_indices = list(range(30)) 
        subset_dataset = torch.utils.data.Subset(original_dataset, selected_indices)

        # create new Dataloader containing the subset
        train_loader1 = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.clients[0].train_loader.batch_size,
            shuffle=False,
            num_workers=self.clients[0].train_loader.num_workers,
            pin_memory=self.clients[0].train_loader.pin_memory
        )
        print(f"Created train_loader1 with {len(subset_dataset)} patches.")'''


    def compute_relevance_and_generate_mask(self):
        """
        Computes relevance and generates the pruning mask.
        """
        print("Starting relevance computation and pruning mask generation.")

        suggested_composite = {
            "low_level_hidden_layer_rule": self.configs["low_level_hidden_layer_rule"],
            "mid_level_hidden_layer_rule": self.configs["mid_level_hidden_layer_rule"],
            "high_level_hidden_layer_rule": self.configs["high_level_hidden_layer_rule"],
            "fully_connected_layers_rule": self.configs["fully_connected_layers_rule"],
            "softmax_rule": self.configs["softmax_rule"],
        }

        print("Layer Rules:")
        print(f"Low-Level Hidden Layer Rule: {self.configs['low_level_hidden_layer_rule']}")
        print(f"Mid-Level Hidden Layer Rule: {self.configs['mid_level_hidden_layer_rule']}")
        print(f"High-Level Hidden Layer Rule: {self.configs['high_level_hidden_layer_rule']}")
        print(f"Fully Connected Layers Rule: {self.configs['fully_connected_layers_rule']}")
        print(f"Softmax Rule: {self.configs['softmax_rule']}")

        if self.configs["model_architecture"] == "vit_b_16":
            composite = get_vit_composite(
                self.configs["model_architecture"], suggested_composite
            )
        else:
            composite = get_cnn_composite(
                self.configs["model_architecture"], suggested_composite
            )

        component_attributor = ComponentAttribution(
            "Relevance",
            "CNN",
            self.layer_types[self.configs["pruning_layer_type"]],
        )

        model_copy = copy.deepcopy(self.model)

        components_relevances = component_attributor.attribute(
            model_copy,
            self.prune_loader,
            composite,
            abs_flag=True,
            device=self.device,
        )

        layer_names = component_attributor.layer_names

        self.pruner = GlobalPruningOperations(
            self.layer_types[self.configs["pruning_layer_type"]],
            layer_names,
        )

        pruning_rate = self.configs.get("pruning_rate", 0.97)

        return self.pruner.generate_global_pruning_mask(
            self.model,
            components_relevances,
            pruning_precentage=pruning_rate,
            subsequent_layer_pruning=self.configs["subsequent_layer_pruning"],
            least_relevant_first=self.configs["least_relevant_first"],
            device=self.device,
        )



    def print_pruning_statistics(self, global_pruning_mask):
        """
        Prints pruning statistics.
        """
        print(f"Pruning-rate: {self.configs.get('pruning_rate', 0.97)}")
        print(f"Scenario: {self.scenario}")
        print("=" * 50)
        print("Layerwise Pruning Rates:")

        total_global_elements = 0
        total_global_zeros = 0  

        for layer, masks in global_pruning_mask.items():
            total_elements = 0
            total_zeros = 0

            for mask_type, mask_values in masks.items():
                if "weight" in mask_values and isinstance(mask_values["weight"], torch.Tensor):
                    tensor = mask_values["weight"]
                    total_elements += tensor.numel()
                    total_zeros += torch.sum(tensor == 0).item()

            percentage_zeros = (total_zeros / total_elements) * 100 if total_elements > 0 else 0
            print(f"Layer: {layer:<20} Num neurons pruned: {total_zeros:<12} % neurons pruned: {percentage_zeros:.2f}%")

            total_global_elements += total_elements
            total_global_zeros += total_zeros

        global_percentage_zeros = (total_global_zeros / total_global_elements) * 100 if total_global_elements > 0 else 0
        print("=" * 50)
        print(f"Overall Percentage of pruned neurons across all layers: {global_percentage_zeros:.2f}%")
        print("=" * 50)


    def distribute_pruning_mask(self, global_pruning_mask):
        """
        Distributes the pruning mask to all clients.
        """
        print("Sending pruning mask to clients...")

        for client in self.clients:
            client.set_pruner_and_mask(
                GlobalPruningOperations(
                    self.layer_types[self.configs["pruning_layer_type"]],
                    self.pruner.layer_names  # Verwende `self.pruner` aus `GlobalClient`
                ),
                global_pruning_mask
            )



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
                labels = batch[4].to(self.device)
                label_new = labels.clone()
                #label_new=np.copy(labels)
            # label_new=self.change_sizes(label_new)

                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()
                predicted_probs += list(probs)

                y_true += list(label_new.cpu().numpy())

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

        # original: update the global model
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
        
        
class PruneDataSet(BENv2DataSet):
    def __init__(self, pruning_patches, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patches = pruning_patches
        self.patches.sort()

    def validate_prune_loader(train_loader, prune_loader, pruning_dataset):
        print("[INFO] Validating the Prune Loader...")

        for batch_idx, (train_batch, prune_batch) in enumerate(zip(train_loader, prune_loader)):
            for i, (train_item, prune_item) in enumerate(zip(train_batch, prune_batch)):
                assert train_item.shape[1:] == prune_item.shape[1:], \
                    f"Shape mismatch in batch {batch_idx}, element {i}: {train_item.shape} != {prune_item.shape}"

        print("[SUCCESS] Prune Loader structure validated!")

def create_prune_loader(train_loader, pruning_patches):
    """
    Creates a Prune Loader based on the train_loader, keeping only the patches
    specified in `pruning_patches`.

    Args:
        train_loader (DataLoader): The original training DataLoader.
        pruning_patches (list): List of patches to be included in the Prune Loader.

    Returns:
        DataLoader: The Prune Loader with the same structure as the train_loader.
    """
    print("[INFO] Creating Prune Loader...")

    train_dataset = train_loader.dataset
    prune_dataset = copy.deepcopy(train_dataset)

    # Filter out patches that are not in pruning_patches
    prune_dataset.patches = [patch for patch in prune_dataset.patches if patch in pruning_patches]
    #print(f"[INFO] {len(prune_dataset.patches)} patches remaining after filtering.")

    # Update dependent attributes
    prune_dataset.BENv2Loader.lbls = {patch: lbl for patch, lbl in prune_dataset.BENv2Loader.lbls.items() if patch in pruning_patches}
    prune_dataset.BENv2Loader.lbl_key_set = set(prune_dataset.patches)

    # Create a new DataLoader based on the filtered dataset
    prune_loader = torch.utils.data.DataLoader(
        dataset=prune_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last,
    )

    print("[SUCCESS] Prune Loader successfully created.")
    return prune_loader