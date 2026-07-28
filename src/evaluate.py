import json
import yaml
import time
import argparse
import torch
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
    train_stats = output_dir / "train_stats.json"

    model_class = MODEL_REGISTRY[args.model]
    model_params = params['models'][args.model]
    model = model_class(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() # Convert into evaluate mode

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0

    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            # per-class accuracy
            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1

                if label == pred:
                    class_correct[label.item()] += 1

    final_loss = test_loss / len(test_loader)
    final_acc = correct / total

    per_class_accuracy = {
        classes[i] : round(class_correct[i] / class_total[i], 2) 
        for i in range(10)
    }

    dummy_batch = torch.randn(1, 3, 32, 32)
    model_cpu = model_class(**model_params)
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            model_cpu(dummy_batch)

    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            model_cpu(dummy_batch)
            times.append(time.perf_counter() - start)

    inference_time_ms = (sum(times) / len(times)) * 1000        

    metrics = {
        "loss"                 : final_loss,
        "accuracy"             : final_acc,
        "inference_time_cpu_ms": inference_time_ms,
        "per_class_accuracy"   : per_class_accuracy
    }

    with open(train_stats, "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats.update(metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(f"-> Evaluation Completed. Acc: {final_acc:.4f}, Loss: {final_loss:.4f}")

if __name__ == "__main__":
    main()

