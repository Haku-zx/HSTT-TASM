# HSTT-TASM: A Hybrid Spatio-Temporal Transformer with Temporal Aggregation and Spatial Memory for Traffic Flow Prediction 

This repository provides the official PyTorch implementation of **HSTT-TASM**, a hybrid spatio-temporal Transformer for traffic flow prediction.  

> Task: multi-step traffic flow forecasting on sensor networks.


## Environment

Tested with a typical modern traffic forecasting PyTorch stack:

- Python >= 3.9
- PyTorch >= 2.0
- CUDA (optional but recommended)

Recommended GPU: NVIDIA GPU with CUDA support.

---

## Installation

### 1) Create a conda environment (recommended)

```bash
conda create -n hstt_tasm python=3.10 -y
conda activate hstt_tasm
````

### 2) Install dependencies

```bash
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas scikit-learn tqdm pyyaml matplotlib
```

---

## Data Preparation

This repo supports the following datasets (traffic sensor networks):

* **PEMS03, PEMS04, PEMS08**
* **METR-LA, PEMS-BAY**

All datasets are commonly used benchmarks for traffic forecasting and are typically processed into
fixed intervals (e.g., 5 minutes) with standard normalization and missing value handling.

### Data directory structure

Put all processed data under `./data/`:

```text
data/
  PEMS03/
  PEMS04/
  PEMS08/
  METR-LA/
  PEMS-BAY/
```

Each dataset folder should contain the processed files required by the code (e.g., adjacency/graph
information and time series tensors).

> Note: This repository assumes data have been preprocessed into the format expected by the dataloader.
> If you uploaded your `data/` folder directly, you can skip additional preprocessing.

---

## Quick Start

### 1) Train

Example (PEMS04):

```bash
python train.py --dataset PEMS04 --history_seq_len 12 --future_seq_len 12 --batch_size 64 --lr 0.001 --gpu 0
```

Common settings used in traffic forecasting:

* Input length: 12 steps
* Output length: 12 steps
* Data split: 6:2:2 (train/val/test)
* Z-score normalization

### 2) Test / Evaluate

```bash
python test.py --dataset PEMS04 --history_seq_len 12 --future_seq_len 12 --gpu 0 --ckpt <PATH_TO_CHECKPOINT>
```

### 3) Metrics

We report standard traffic forecasting metrics:

* MAE
* RMSE
* MAPE (%)

---

## Reproducibility Tips

* Fix random seeds for Python/NumPy/PyTorch
* Use the same dataset split protocol (train/val/test)
* Ensure the same sampling interval (e.g., 5-min)
* Keep consistent normalization statistics (fit on training set only)

---

## Project Structure (Typical)

```text
.
├── data/                  # datasets (processed)
├── train.py               # training entry
├── test.py                # evaluation entry
├── model.py               # HSTT-TASM model definition
├── util.py                # utilities (logging/metrics/seed/etc.)
├── configs/               # optional: yaml configs
└── README.md
```


## Acknowledgements

This project builds upon common practices and benchmarks in spatiotemporal traffic forecasting research.

