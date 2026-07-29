import os
import json
import glob
import yaml
from datetime import date
from statistics import mean
from config import PARAMS, ROOT

def load_all_metrics() -> dict:
    metrics_paths = (ROOT / "experiments").glob("*/metrics.json")
    
    metrics_dict = {}
    for path in metrics_paths:
        with open(path, "r") as f:
            metrics = json.load(f)
        
        model = path.parent.name  
        metrics_dict[model] = metrics
    
    return metrics_dict

def find_hardest_classes(metrics: dict) -> dict:
    for model in metrics.keys():
        per_class_accuracy = metrics[model]['per_class_accuracy']
        hardest_class = min(per_class_accuracy, key=per_class_accuracy.get)
        metrics[model]['hardest_class'] = hardest_class

    return metrics

def generate_markdown(metrics: dict) -> str:
    with open(PARAMS, "r") as f:
        params = yaml.safe_load(f)

    models = sorted(
        metrics.items(),
        key=lambda x: x[1]["num_params"]
    )

    seed = params['base']['seed']
    epochs = params['base']['epochs']
    # ---------------- Summary table ----------------
    md = f"""# CNN Benchmark Report — CIFAR-10

*Generated: {date.today()} | Seed: {seed} | Epochs: {epochs}*

## Summary

| Model | Accuracy | Parameters | Inference Time (ms) |
|------|---------:|-----------:|--------------------:|
"""

    for model, result in models:
        md += (
            f"| {model} | "
            f"{result['accuracy']:.2%} | "
            f"{result['num_params']:,} | "
            f"{result['inference_time_cpu_ms']:.3f} |\n"
        )

    # ---------------- Key Findings ----------------
    md += "\n## Key Findings\n\n"

    md += "**Accuracy vs. Parameters**\n\n"

    for (name1, m1), (name2, m2) in zip(models, models[1:]):
        md += (
            f"- Scaling from **{name1}** to **{name2}** improves accuracy "
            f"by **{m2['accuracy'] - m1['accuracy']:.1%}** while requiring "
            f"**{m2['num_params'] / m1['num_params']:.1f}×** more parameters.\n"
        )

    # ---------------- Inference trade-off ----------------
    if len(models) >= 2:
        prev_name, prev = models[-2]
        largest_name, largest = models[-1]

        speed_ratio = (
            largest["inference_time_cpu_ms"]
            / prev["inference_time_cpu_ms"]
        )

        acc_diff = (
            largest["accuracy"]
            - prev["accuracy"]
        )

        md += (
            f"\n**Inference Trade-off:** Compared with **{prev_name}**, "
            f"**{largest_name}** is **{speed_ratio:.1f}×** slower while "
            f"improving accuracy by only **{acc_diff:.1%}**.\n"
        )

    # ---------------- Hardest class ----------------
    classes = next(iter(metrics.values()))["per_class_accuracy"].keys()

    avg_class_acc = {
        cls: mean(
            model["per_class_accuracy"][cls]
            for model in metrics.values()
        )
        for cls in classes
    }

    hardest_class = min(avg_class_acc, key=avg_class_acc.get)

    md += (
        f"\n**Hardest Class:** `{hardest_class}` has the lowest average "
        f"accuracy across all models "
        f"(**{avg_class_acc[hardest_class]:.1%}**), suggesting that this "
        "category is inherently more difficult to classify due to greater "
        "visual similarity with other classes.\n"
    )

    # ---------------- Detailed results ----------------
    md += "\n## Detailed Results\n\n"

    for model, result in models:
        md += f"### {model}\n\n"

        md += "| Metric | Value |\n"
        md += "|-------|------:|\n"
        md += f"| Accuracy | {result['accuracy']:.2%} |\n"
        md += f"| Loss | {result['loss']:.4f} |\n"
        md += f"| Parameters | {result['num_params']:,} |\n"
        md += (
            f"| Training Time | "
            f"{result['training_time_minutes']} min |\n"
        )
        md += (
            f"| Inference Time | "
            f"{result['inference_time_cpu_ms']:.3f} ms |\n"
        )
        md += f"| Hardest Class | {result['hardest_class']} |\n\n"

        md += "| Class | Accuracy |\n"
        md += "|------|---------:|\n"

        for cls, acc in sorted(
            result["per_class_accuracy"].items(),
            key=lambda x: x[1]
        ):
            md += f"| {cls} | {acc:.2%} |\n"

        md += "\n"

    judgment = """
## Conclusion

- **Best accuracy:** convnext_v1 (86.34%)
- **Best speed:** lenet (0.428ms) 
- **Best trade-off:** pintee_cnn — 80.84% accuracy với inference 
  nhanh hơn convnext 4× và chỉ 8.8M params
- **Consistent weak spot:** `dog` và `cat` khó nhất với tất cả models
"""

    md += judgment

    return md

def main():    
    metrics = load_all_metrics()
    metrics = find_hardest_classes(metrics)
    report = generate_markdown(metrics)

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    with open(report_dir / "benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()