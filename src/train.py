import os
import csv
import yaml
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import PinteeCNN
from src.config import MODEL, PARAMS, TRAIN_DATA, TRAINING_LOG

torch.serialization.add_safe_globals([torchvision.datasets.cifar.CIFAR10])

with open(PARAMS, "r") as f:
    params = yaml.safe_load(f)

def main():
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

    # Load improvements
    if params['improvements']['reduce_learning_rate']:
        lr = 1e-4

    # Initialize model, optimizer, loss
    model = PinteeCNN(n_classes=10).to(device)
    optimizer = Adam(model.parameters(), lr=lr)  
    criterion = nn.CrossEntropyLoss()

    with open(TRAINING_LOG, mode='w', newline="") as f:
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

    # Save model
    torch.save(model.state_dict(), MODEL)
    print(f"-> Saved to {MODEL}")

if __name__ == "__main__":
    main()