import torch

def check_gpu():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
    else:
        print("No GPU detected!")

if __name__ == "__main__":
    check_gpu()
