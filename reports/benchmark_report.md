# CNN Benchmark Report — CIFAR-10

*Generated: 2026-07-29 | Seed: 42 | Epochs: 20*

## Summary

| Model | Accuracy | Parameters | Inference Time (ms) |
|------|---------:|-----------:|--------------------:|
| lenet | 63.24% | 82,838 | 0.428 |
| pintee_cnn | 80.84% | 8,873,994 | 2.794 |
| convnext_v1 | 86.34% | 19,071,562 | 11.332 |

## Key Findings

**Accuracy vs. Parameters**

- Scaling from **lenet** to **pintee_cnn** improves accuracy by **17.6%** while requiring **107.1×** more parameters.
- Scaling from **pintee_cnn** to **convnext_v1** improves accuracy by **5.5%** while requiring **2.1×** more parameters.

**Inference Trade-off:** Compared with **pintee_cnn**, **convnext_v1** is **4.1×** slower while improving accuracy by only **5.5%**.

**Hardest Class:** `dog` has the lowest average accuracy across all models (**62.0%**), suggesting that this category is inherently more difficult to classify due to greater visual similarity with other classes.

## Detailed Results

### lenet

| Metric | Value |
|-------|------:|
| Accuracy | 63.24% |
| Loss | 1.0227 |
| Parameters | 82,838 |
| Training Time | 2 min |
| Inference Time | 0.428 ms |
| Hardest Class | dog |

| Class | Accuracy |
|------|---------:|
| dog | 44.00% |
| bird | 46.00% |
| cat | 51.00% |
| horse | 63.00% |
| airplane | 64.00% |
| deer | 65.00% |
| truck | 68.00% |
| automobile | 77.00% |
| frog | 77.00% |
| ship | 77.00% |

### pintee_cnn

| Metric | Value |
|-------|------:|
| Accuracy | 80.84% |
| Loss | 0.5840 |
| Parameters | 8,873,994 |
| Training Time | 7 min |
| Inference Time | 2.794 ms |
| Hardest Class | dog |

| Class | Accuracy |
|------|---------:|
| dog | 60.00% |
| cat | 75.00% |
| airplane | 78.00% |
| bird | 78.00% |
| deer | 81.00% |
| horse | 84.00% |
| truck | 85.00% |
| frog | 86.00% |
| automobile | 87.00% |
| ship | 95.00% |

### convnext_v1

| Metric | Value |
|-------|------:|
| Accuracy | 86.34% |
| Loss | 0.4708 |
| Parameters | 19,071,562 |
| Training Time | 35 min |
| Inference Time | 11.332 ms |
| Hardest Class | cat |

| Class | Accuracy |
|------|---------:|
| cat | 73.00% |
| dog | 82.00% |
| deer | 83.00% |
| bird | 84.00% |
| frog | 86.00% |
| horse | 87.00% |
| airplane | 91.00% |
| ship | 91.00% |
| truck | 91.00% |
| automobile | 95.00% |


## Conclusion

- **Best accuracy:** convnext_v1 (86.34%)
- **Best speed:** lenet (0.428ms) 
- **Best trade-off:** pintee_cnn — 80.84% accuracy với inference 
  nhanh hơn convnext 4× và chỉ 8.8M params
- **Consistent weak spot:** `dog` và `cat` khó nhất với tất cả models
