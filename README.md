# TimeGEN

> **TimeGEN: A Cross-Domain and Generative Model for Time Series Forecasting**  

## Abstract

We propose TimeGEN, an MLP-based generative deep learning architecture for Transfer Learning in time series forecasting. We focus on cross-domain heterogeneity, where training and test domains may differ in scale, sampling frequency, seasonality, sparsity, volatility, or local temporal dynamics. TimeGEN combines temporal normalization, a shared variational encoder, and a lightweight residual multiscale decoder to learn transferable forecasting structure across heterogeneous domains. The encoder maps each normalized input window to a stochastic context in a shared latent space, and this context is provided to every residual decoder block. This design differs from conditioning on observed exogenous variables or injecting context only at the input layer. TimeGEN further supervises the backcast path with an explicit reconstruction loss, jointly optimized with forecasting loss and KL regularization, so the latent context must support both input reconstruction and future prediction.

Extensive empirical results across ten public datasets show that TimeGEN achieves the best average performance in the single-source and multi-source out-of-domain settings while remaining competitive in full-shot and in-domain transfer settings. In the multi-source out-of-domain setting, TimeGEN reduces forecasting error by 6.4%–34.7% relative to competing SOTA methods, while achieving a 1.3–30× speedup in training time compared to SOTA MLP and Transformer methods.

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
- **Fast Training**: 1.3–30× faster than SOTA alternatives
- **Zero-shot Generalization** to unseen time series and domains 

---

## Results Summary

MASE and average rank (lower is better) across datasets and evaluation settings. MASE values are unweighted means across the ten dataset-frequency combinations reported in the appendix, and average rank is computed across the same ten benchmarks. Best and second-best values are **bolded** and _underlined_. The *Time* column reports normalized training time relative to the fastest method (TimeGEN = 1.0).

| Method | Time (× TimeGEN) | Full-shot MASE | Rank | In-domain MASE | Rank | Single-source MASE | Rank | Multi-source MASE | Rank |
|--------|------------------|----------------|------|----------------|------|---------------------|------|-------------------|------|
| **TimeGEN** | 1.0 | 1.332 | 3.6 | 1.409 | 5.4 | **2.076** | **1.9** | **1.493** | **2.4** |
| KAN | 2.076 | 1.376 | 4.6 | 1.442 | 4.8 | 8.060 | 6.0 | 1.627 | 4.9 |
| NBEATS | 1.307 | **1.315** | **3.2** | **1.245** | **2.4** | 2.287 | 5.7 | 2.001 | 5.2 |
| NHITS | 3.507 | _1.319_ | _3.4_ | 1.423 | _4.0_ | 86.653 | 7.3 | 1.783 | 6.3 |
| PatchTST | 5.235 | 1.432 | 6.4 | 1.438 | 6.2 | _2.128_ | _3.0_ | 1.681 | 5.8 |
| TFT | 4.105 | 1.439 | 6.0 | 1.474 | 6.3 | 2.280 | 5.0 | 1.714 | 5.4 |
| TSMixer | 31.042 | 1.810 | 9.7 | 1.675 | 9.0 | 2.298 | 5.7 | 1.816 | 8.2 |
| TimeMOE | 28.082 | 1.445 | 6.6 | _1.386_ | 6.0 | 2.437 | 7.2 | 1.837 | 7.2 |
| iTransformer | 26.760 | 1.667 | 9.5 | 1.667 | 8.9 | 2.552 | 8.5 | 2.221 | 6.7 |
| xLSTM | 2.821 | 1.493 | 6.1 | 1.905 | 7.0 | 4.524 | 9.3 | _1.595_ | _4.3_ |
| TimeMixer | 1.349 | 1.442 | 7.1 | 1.423 | 5.7 | 2.296 | 6.4 | 1.888 | 8.7 |

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
