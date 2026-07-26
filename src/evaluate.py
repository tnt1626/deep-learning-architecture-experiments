import json
import torch
import yaml
import argparse
import torchvision
from torch.utils.data import DataLoader
from config import ROOT, PARAMS, TEST_DATA
from registry import MODEL_REGISTRY

torch.serialization.add_safe_globals([torchvision.datasets.cifar.CIFAR10])

with open(PARAMS, "r") as f:
    params = yaml.safe_load(f)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_dataset = torch.load(TEST_DATA, weights_only=False, map_location=device)
    test_loader = DataLoader(test_dataset, batch_size=params['base']['batch_size'], shuffle=False)

    # Load trained model 
    parser = argparse.ArgumentParser(description="Get model from MODEL_REGISTRY.")
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    args = parser.parse_args()

    output_dir = ROOT / "experiments" / args.model
    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "model.pth"

    model_class = MODEL_REGISTRY[args.model]
    model_params = params['models'][args.model]
    model = model_class(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() # Convert into evaluate mode

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    final_loss = test_loss / len(test_loader)
    final_acc = correct / total

    metrics = {
        "loss": final_loss,
        "accuracy": final_acc
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"-> Evaluation Completed. Acc: {final_acc:.4f}, Loss: {final_loss:.4f}")

if __name__ == "__main__":
    main()

