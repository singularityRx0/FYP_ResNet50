import torch

def gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Model training on GPU....")
    else:
        device = torch.device("cpu")
        print("Model training on CPU....")

    return device


if __name__ == '__main__':
    gpu()
