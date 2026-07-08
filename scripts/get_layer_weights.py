import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the weights of a specific layer from a PyTorch model.")
    parser.add_argument("--model_path", type=str, help="Path to the PyTorch model file (e.g., .pt or .pth).")
    args = parser.parse_args()

    # Load the model
    model = torch.load(args.model_path)

    for name, param in model.items():
        if "layer_logits" in name:
            print(f"Layer: {name},  weights_softmax, {param} ,Weights: {torch.softmax(param, dim=0)}")