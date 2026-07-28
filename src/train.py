import os
import csv
import yaml
import time
import json
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from config import PARAMS, ROOT, TRAIN_DATA
from registry import MODEL_REGISTRY

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

torch.serialization.add_safe_globals([torchvision.datasets.cifar.CIFAR10])

with open(PARAMS, "r") as f:
    params = yaml.safe_load(f)

def main():
    start = time.time()

    seed = params['base']['seed']
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create folder models if not exist
    os.makedirs("models", exist_ok=True)

    # Load base configuration
    epochs = params['base']['epochs']
    batch_size = params['base']['batch_size']
    lr = params['base']['learning_rate']

    # Load dataset
    train_dataset = torch.load(TRAIN_DATA, weights_only=False, map_location=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, loss
    ## Get model name argument
    parser = argparse.ArgumentParser(description="Get model from MODEL_REGISTRY.")
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    args = parser.parse_args()

    model_class = MODEL_REGISTRY[args.model]
    model_params = params["models"][args.model]
    model = model_class(**model_params).to(device)

    optimizer = Adam(model.parameters(), lr=lr)  
    criterion = nn.CrossEntropyLoss()

    output_dir = ROOT / "experiments" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log = output_dir / "training_log.csv"
    train_stats = output_dir / "train_stats.json"
    model_path = output_dir / "model.pth"

    with open(training_log, mode='w', newline="") as f:
        writer = csv.writer(f)
        # Write header row to CSV file
        writer.writerow(["epoch", "loss", "accuracy"])

        # Loop over all epoch
        for epoch in range(epochs):
            model.train() # Set model to training mode
            running_loss, correct, total = 0.0, 0, 0

            # Iterate over training data
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad() # Reset gradients from previous step
                outputs = model(inputs) # Forward pass
                loss = criterion(outputs, labels) # Compute loss
                loss.backward() # Backpropagation
                optimizer.step() # Update model parameters

                # Accumulate loss
                running_loss += loss.item()

                # Get predicted class (max logit)
                _, predicted = outputs.max(1)

                # Update total samples
                total += labels.size(0)

                # Count correct predictions
                correct += predicted.eq(labels).sum().item()

            # Compute average loss for the epoch
            epoch_loss = running_loss / len(train_loader)

             # Compute accuracy for the epoch
            epoch_acc = correct / total
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
            writer.writerow([epoch, epoch_loss, epoch_acc])

    training_time = (time.time() - start) / 60
    stats = {
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_time_minutes": round(training_time)
    }
    with open(train_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"-> Saved to {model_path}")

if __name__ == "__main__":
    main()