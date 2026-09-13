| Method | Optimizer | Initial Learning Rate | Scheduler | Batch Size | Epochs | Weight Decay |
|---|---|---:|---|---:|---:|---:|
| CLIP | N/A | N/A | N/A | N/A | 0 | N/A |
| CoOp | SGD | 0.002 | Cosine | 64 | 150 | 5×10⁻⁴ |
| CoCoOp | SGD | 0.002 | Cosine | 1 | 10 | 5×10⁻⁴ |
| MaPLe | SGD | 0.0035 | Cosine | 4 | 5 | 5×10⁻⁴ |
| Tip-Adapter-F | AdamW | 0.001 | Cosine annealing | 256 | 20 | 0.01 |
| DSRA | SGD | 0.002 | Cosine | 64 | 150 | 5×10⁻⁴ |
