# Soil and Climate-Driven Crop Recommendation via Ensemble Learning

A comparative study of eight classical machine-learning classifiers and a heterogeneous
soft-voting ensemble for crop recommendation from soil and climatic attributes, with an
emphasis on stability and resource-constrained (edge/mobile) deployment.

This repository contains the code accompanying the paper:

> **Soil and Climate-Driven Crop Recommendation via Ensemble Learning: A Comparative Study of
> Classical Classifiers for Resource-Constrained Deployment.**
> C. Maji, P. Pal, R. K. Singh, and S. Upadhyay.
> *Proc. International Conference on Intelligent Systems and Robotics for Sustainable Development
> (ISRSD-2026)*, Springer Lecture Notes in Electrical Engineering (LNEE). _In press._
> DOI: _to be added on publication._

Department of Computer Science and Engineering, School of Engineering,
Dayananda Sagar University (DSU), Bengaluru, India.

---

## Overview

Choosing a suitable crop for a given combination of soil nutrients and climate is a recurring
decision for small and marginal farmers. Using a benchmark dataset of **2,200 samples across
22 crop classes** described by seven features (N, P, K, temperature, humidity, soil pH, rainfall),
this work:

1. Compares eight classical classifiers under a single, transparent protocol
   (identical preprocessing, stratified 80/20 split, default hyperparameters).
2. Proposes a **heterogeneous soft-voting ensemble** (Random Forest + Gaussian Naive Bayes +
   XGBoost + Gradient Boosting) and evaluates its stability across ten random seeds.
3. Benchmarks the ensemble against a homogeneous bagging baseline.
4. Profiles every model for **on-device deployment** (serialized size and inference latency).

## Key results

| Model | Accuracy | Weighted F1 |
|-------|:--------:|:-----------:|
| Random Forest | 0.9955 | 0.9955 |
| **Proposed Ensemble** | **0.9955** | **0.9955** |
| Gaussian Naive Bayes | 0.9955 | 0.9954 |
| XGBoost | 0.9932 | 0.9931 |
| Gradient Boosting | 0.9886 | 0.9887 |
| SVM (RBF) | 0.9841 | 0.9840 |
| Decision Tree | 0.9795 | 0.9794 |
| KNN | 0.9795 | 0.9793 |
| Logistic Regression | 0.9727 | 0.9725 |

- **Stability:** across ten random seeds the proposed ensemble has the lowest accuracy standard
  deviation (0.0021) — a 19–28% reduction versus the best individual classifiers and a further
  21% versus a homogeneous bagging baseline.
- **Deployment:** Gaussian Naive Bayes retains top accuracy (99.55%) at only ~3.4 KB, while every
  model answers a single query in under ~14 ms on a commodity CPU.

## Dataset

This work uses the publicly available **Crop Recommendation Dataset** (A. Ingle, Kaggle, 2020):
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

The dataset is **not redistributed here** — please download `Crop_recommendation.csv` from the
Kaggle link above and place it in the repository root (or update the path in the notebook).

## Repository structure

```
.
├── IEEE_Paper2_Crop_Recommendation_FULL.ipynb   # end-to-end pipeline + figures
├── requirements.txt                              # Python dependencies
├── README.md
└── .gitignore
```

## Getting started

```bash
# 1. clone
git clone https://github.com/Chandan460/crop-recommendation-ensemble.git
cd crop-recommendation-ensemble

# 2. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. download Crop_recommendation.csv from Kaggle (link above) into this folder

# 5. run the notebook
jupyter notebook IEEE_Paper2_Crop_Recommendation_FULL.ipynb
```

## Reproducibility

- All splits use stratified sampling with a fixed random seed of **42**.
- Tree-based learners use raw features; distance/margin-based learners use standardized features
  (`StandardScaler` fit on the training partition only).
- Experiments were run on a single CPU (Google Colaboratory backend); the full pipeline completes
  in under three minutes.

**Note on exact figures.** Classification accuracies are deterministic given the fixed seed and
reproduce the paper for all models except XGBoost, whose accuracy varies slightly with the XGBoost
version (the paper used 1.5.x; newer versions report ~0.991 vs the paper's 0.9932). Inference
latency in the deployment profile is hardware-dependent, so absolute milliseconds will differ from
the paper's Google Colaboratory measurements; model sizes (KB) and the relative ordering are
consistent across machines.

## Citation

```bibtex
@inproceedings{maji2026crop,
  title     = {Soil and Climate-Driven Crop Recommendation via Ensemble Learning:
               A Comparative Study of Classical Classifiers for Resource-Constrained Deployment},
  author    = {Maji, Chandan and Pal, Pratik and Singh, Rohit Kumar and Upadhyay, Shubham},
  booktitle = {Proc. International Conference on Intelligent Systems and Robotics for
               Sustainable Development (ISRSD-2026)},
  series    = {Lecture Notes in Electrical Engineering},
  publisher = {Springer},
  year      = {2026},
  note      = {In press}
}
```

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The dataset is subject to its own license on Kaggle and is not included here.
