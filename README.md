# CNN Architecture Benchmark - CIFAR-10

Reproducible benchmark for comparing three CNN architectures on CIFAR-10 with a DVC pipeline. Every experiment is tracked and can be reproduced with a single `dvc repro` run.

## Benchmark Summary

| Model | Accuracy | Params | Inference |
|---|---:|---:|---:|
| LeNet-5 | 63.24% | 82,838 | 0.428 ms |
| PinteeCNN | 80.84% | 8,873,994 | 2.794 ms |
| ConvNeXt v1 | 86.34% | 19,071,562 | 11.332 ms |

Full results and class-level analysis are available in [reports/benchmark_report.md](reports/benchmark_report.md).

## Key Takeaways

- LeNet-5 is the fastest and smallest model.
- PinteeCNN gives the best balance between accuracy, size, and inference speed.
- ConvNeXt v1 achieves the highest accuracy, but at a much higher parameter and latency cost.

## Design Decisions

- **Fair comparison:** all models trained with identical hyperparameters 
  (lr, epochs, batch size, seed) — only architecture differs.
- **Data augmentation:** RandomHorizontalFlip + RandomCrop applied to 
  all models to reduce overfitting without favoring any architecture.
- **ConvNeXt adaptation:** stem modified from stride=4 to stride=1 for 
  CIFAR-10's 32×32 input. Original design targets 224×224 (ImageNet).
- **Inference time:** measured on CPU with batch size=1 and 10-run warmup, 
  averaged over 100 runs for stability.

## Reproduce

```bash
git clone <repo-url>
cd deep-learning-architecture-experiments
dvc pull
dvc repro
dvc metrics show
```

## DVC Pipeline

The pipeline is split into four reproducible stages:

- `prepare_data` - download and preprocess CIFAR-10
- `train_lenet`, `train_pintee_cnn`, `train_convnext_v1` - train each architecture
- `evaluate_lenet`, `evaluate_pintee_cnn`, `evaluate_convnext_v1` - compute metrics for each run
- `benchmark` - generate the comparison report in `reports/benchmark_report.md`

## Project Layout

```text
src/                 training, evaluation, benchmarking, and data prep scripts
src/architectures/   model definitions for the CNN experiments
data/                raw and processed CIFAR-10 data managed by DVC
experiments/         per-model checkpoints, logs, and metrics
reports/             benchmark report and summary outputs
dvc.yaml             end-to-end pipeline definition
params.yaml          shared configuration for training and model variants
```

## Notes

- Model behavior and hyperparameters are controlled through `params.yaml`.
- All generated artifacts are managed by DVC, so the benchmark can be rebuilt from scratch on a clean checkout.
