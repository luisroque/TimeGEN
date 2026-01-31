# TimeGEN

> **TimeGEN: A Cross-Domain and Generative Model for Time Series Forecasting**  

## Abstract

We propose TimeGEN, a lightweight, MLP-based generative deep learning architecture for Transfer Learning in time series forecasting. We use a variational encoder to capture high-level temporal representations across diverse series and domains. To further strengthen this generalization, we combine a reconstruction and forecasting loss, which shapes the latent space to retain local detail while capturing global predictive dependencies. In addition, temporal normalization ensures robustness to varying input scales and noise. To capture multiscale dynamics, we integrate a modular decoder that combines neural basis expansion with multi-rate interpolation, balancing long-range trends with high-frequency variations. Extensive empirical results across ten public datasets demonstrate that TimeGEN consistently outperforms SOTA methods in zero-shot and cross-domain settings. In cross-domain settings, TimeGEN reduces forecasting error by more than 8\% and up to 38\%, while achieving a 2-30x speedup in training time compared to SOTA MLP and Transformer methods.

---

## Repository Structure


```
TimeGEN/
│
├── assets/              # Weights, results, etc
├── timegen/             # Core package
│   ├── data_pipeline/       # Data loading and preprocessing
│   ├── experiments/         # Experiment orchestration and helpers
│   ├── load_data/           # Dataset-specific loading functions
│   ├── metrics/             # Evaluation metrics
│   ├── model_pipeline/      # TimeGEN and baseline model implementations
│   ├── visualization/       # Plotting tools and analysis
├── requirements.txt
└── README.md
```


---

## Key Features

- **Variational Encoder** for latent temporal representations  
- **Modular MLP Decoder** with multi-rate pooling and basis expansion  
- **Temporal Normalization** for domain shift robustness  
- **Fast Training**: 2–30× faster than SOTA alternatives
- **Zero-shot Generalization** to unseen time series and domains 

---

## Results Summary

MASE and average rank (lower is better) across datasets and evaluation settings. Best and second-best values are **bolded** and _underlined_. The *Time* column reports normalized training time relative to the fastest method (TimeGEN = 1.0).

| Method | Time (× TimeGEN) | Full-shot MASE | Rank | In-domain MASE | Rank | Single-source MASE | Rank | Multi-source MASE | Rank |
|--------|------------------|----------------|------|----------------|------|---------------------|------|-------------------|------|
| **TimeGEN** | 1.0 | _1.332_ | _2.7_ | **1.409** | 3.8 | **1.926** | **3.0** | **1.493** | **1.8** |
| KAN | 2.076 | 1.376 | 3.3 | 1.442 | _3.2_ | 10.962 | 4.7 | _1.627_ | _3.5_ |
| NHITS | 3.507 | **1.319** | **2.3** | _1.423_ | **2.8** | 161.73 | 5.0 | 1.783 | 4.6 |
| PatchTST | 5.235 | 1.432 | 4.6 | 1.438 | 4.4 | _1.986_ | _3.5_ | 1.681 | 4.2 |
| TFT | 4.105 | 1.439 | 4.1 | 1.474 | 4.3 | 2.163 | 3.9 | 1.714 | 4.2 |
| TSMixer | 31.042 | 1.810 | 7.2 | 1.676 | 6.5 | 2.160 | 5.0 | 1.816 | 6.2 |
| TimeMOE | 28.082 | 1.445 | 4.7 | 1.443 | 4.2 | 2.255 | 5.1 | 1.837 | 5.4 |
| iTransformer | 26.760 | 1.667 | 7.0 | 1.632 | 6.4 | 2.429 | 5.4 | 2.416 | 5.1 |

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
