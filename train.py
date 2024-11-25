import torch
import argparse
from models.pytorch_models import ResNet50
from utils.clients import GlobalClient
from utils.pytorch_utils import start_cuda

def parse_args():
    """
    Parses command line arguments for training configurations.
    """
    parser = argparse.ArgumentParser(description="Train a federated learning model with optional pruning.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for client training.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds.")
    parser.add_argument("--pruning", action="store_true", help="Enable pruning functionality.")
    parser.add_argument("--pruning_rate", type=float, default=0.3, help="Pruning rate if pruning is enabled.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for training: 'cuda' or 'cpu'.")
    parser.add_argument("--data_dirs", type=dict, default=None, help="Dictionary for dataset directories.")
    args = parser.parse_args()
    return args

def main():
    """
    Main function for initializing and executing federated learning.
    """
    args = parse_args()

    # Start CUDA if available
    device = start_cuda() if args.device == "cuda" else torch.device("cpu")

    # Initialize model
    model = ResNet50(num_classes=19)

    # Define data directories (update as needed)
    data_dirs = {
        "images_lmdb": "/path/to/images.lmdb",
        "metadata_parquet": "/path/to/metadata.parquet",
        "metadata_snow_cloud_parquet": "/path/to/metadata_snow_cloud.parquet",
    }

    # Create GlobalClient for federated learning
    global_client = GlobalClient(
        model=model,
        lmdb_path=data_dirs["images_lmdb"],
        val_path=data_dirs["metadata_parquet"],
        csv_paths=["Finland", "Ireland", "Serbia"],  # Replace with actual CSV paths
        batch_size=args.batch_size,
        num_classes=19,
        data_dirs=data_dirs,
        device=device,
    )

    # Train the model
    print(f"Starting federated learning with {args.rounds} rounds and {args.epochs} epochs per client...")
    results, client_results = global_client.train(communication_rounds=args.rounds, epochs=args.epochs)

    # Save the final model and results
    global_client.save_state_dict()
    global_client.save_results()
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
