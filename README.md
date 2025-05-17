# TimeGEN

> **TimeGEN: An Efficient Generative Foundation Model for Time Series Forecasting**  
> NeurIPS 2025 Submission

---

## Overview

**TimeGEN** is a lightweight generative foundation model for time series forecasting, designed to generalize across tasks and domains without relying on attention mechanisms or large-scale pretraining. It combines a variational encoder with a modular MLP-based decoder, providing competitive performance and superior training efficiency in both full-shot and zero-shot settings.

---

## Repository Structure


```
TimeGEN/
│
├── assets/              # Weights, results, etc
├── lightning_logs/      # Checkpoints and logs (Lightning)
├── timegen/             # Core package
│   ├── data_pipeline/       # Data loading and preprocessing
│   ├── experiments/         # Experiment orchestration and helpers
│   ├── load_data/           # Dataset-specific loading functions
│   ├── metrics/             # Evaluation metrics
│   ├── model_pipeline/      # TimeGEN and baseline model implementations
│   ├── visualization/       # Plotting tools and analysis
│   └── __init__.py
├── requirements.txt
├── LICENSE
└── README.md
```


---

## Key Features

- **Variational Encoder** for latent temporal representations  
- **Modular MLP Decoder** with multi-rate pooling and basis expansion  
- **Reversible Instance Normalization (RevIN)** for domain shift robustness  
- **Fast Training**: 5–30× faster than Transformer-based alternatives  
- **Probabilistic Forecasting** via latent sampling  
- **Zero-shot Generalization** to unseen time series and domains  

---

## Results Summary

Below is the main table from the paper, comparing TimeGEN to other baselines across four evaluation regimes:

| Method         | Params (M) | Rel. Time | Full-shot MASE | Rank | In-domain MASE | Rank | Single-source MASE | Rank | Multi-source MASE | Rank |
|----------------|-------------|-----------|------------------|------|------------------|------|----------------------|------|---------------------|------|
| **TimeGEN**     | 2.7         | 1.0       | 1.33             | 4.6  | 1.41             | 5.8  | **1.93**             | **3.8**  | **1.55**             | **2.4**  |
| TimeGEN-S       | —           | 1.0       | **1.31**         | 3.8  | **1.32**         | **2.8**  | 2.50                 | 5.6  | —                   | —    |
| TimeGEN-D       | —           | 1.5       | _1.32_           | **3.2**  | 1.34             | 3.6  | 5.97                 | 5.7  | —                   | —    |
| TimeGEN-M       | 2.8         | 0.7       | 1.33             | 3.6  | _1.33_           | _3.4_  | 2.72                 | _4.5_  | 1.98                 | _2.5_  |
| KAN             | —           | 2.2       | 1.37             | 5.2  | 1.44             | 5.4  | 11.12                | 6.2  | —                   | —    |
| NHITS           | —           | 3.7       | _1.32_           | _3.4_  | 1.42             | 4.3  | 166.95               | 6.6  | —                   | —    |
| PatchTST        | 0.9         | 5.2       | 1.43             | 6.9  | 1.44             | 6.4  | _1.99_               | 4.5  | 1.62                 | 3.5  |
| TFT             | 2.1         | 4.2       | 1.44             | 6.1  | 1.47             | 6.2  | 2.17                 | 5.0  | 1.70                 | 4.1  |
| TSMixer         | —           | 32.5      | 1.81             | 9.1  | 1.68             | 8.6  | 2.16                 | 6.3  | —                   | —    |
| iTransformer    | —           | 28.2      | 1.67             | 9.1  | 1.67             | 8.5  | 2.43                 | 6.7  | —                   | —    |
| Moirai-small    | 14          | —         | —                | —    | —                | —    | —                    | —    | _1.61_               | 3.3  |
| TimeMOE-base    | 50          | —         | —                | —    | —                | —    | —                    | —    | 1.83                 | 5.2  |

---

## Datasets

TimeGEN is evaluated on 10 publicly available datasets including:

- **M1, M3, M4, M5** forecasting competitions
- **Tourism** dataset
- **Traffic** from SF Bay Area

Spanning monthly, quarterly, yearly, and daily frequencies, totaling over **100k series and 60M+ observations**.

---

## Running Experiments

> To reproduce key experiments with **TimeGEN**, use the appropriate command below based on the selected evaluation regime.


- **Multi-source Out-Domain Transfer:**
  ```bash
  python timegen/experiments/run_pipeline.py --use-gpu --coreset
  ```
- **Single-source Transfer (In-domain and Out-domain):**
  ```bash
  python timegen/experiments/run_pipeline.py --use-gpu --transfer-learning
  ```
- **Full-shot Forecasting:**
  ```bash
  python timegen/experiments/run_pipeline.py --use-gpu --basic-forecasting
  ```
