# Comparative Performance Analysis of Supervised Machine Learning Algorithms for Precision Crop Recommendation: Investigating Split Variance, Dataset Separability, and Model Collapse

**Puppala Anjali**$^{1}$, **Kasarla Vijitha**$^{1}$  
$^{1}$Department of Information Technology, Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad, India  
*Correspondence:* ugs24310_inf.kasarla@cbit.org.in  

---

## Abstract
In recent smart-agriculture literature, machine learning (ML) models for soil- and climate-driven crop recommendation routinely report near-ceiling classification accuracies exceeding 98% to 99%. However, the vast majority of these studies evaluate predictive models under an arbitrary single 80/20 train-test holdout without reporting split-to-split variance, confidence bounds, or cross-dataset validation. Consequently, it remains unresolved whether these reported near-perfect scores reflect genuine algorithmic capability or are artifacts of favorable split selection and highly separable benchmark data. To address this methodological gap, this study conducts an exhaustive, controlled comparative benchmarking audit across five supervised machine learning algorithm families: Random Forest (RF), Extreme Gradient Boosting (XGBoost), Support Vector Machine with Radial Basis Function kernel (SVM-RBF), K-Nearest Neighbors (KNN), and Gaussian Naive Bayes (GNB). 

All algorithms were configured using frozen hyperparameter baselines and evaluated under a rigorous zero-leakage preprocessing protocol across three matched validation frameworks: (1) an unstratified single 80/20 holdout split, (2) stratified 5-fold cross-validation, and (3) a 10× repeated stratified holdout across independent random seeds. Experiments were conducted on two agricultural datasets: the widely cited Kaggle Crop Recommendation benchmark (2,200 samples, 22 classes) and a secondary smart farming IoT sensor dataset (500 samples, 5 classes). 

Our empirical results expose a critical divergence: while all models reproduce ~98.1% to 99.8% accuracy on Dataset 1, every model experiences catastrophic performance collapse on Dataset 2, scoring between 17.0% and 23.3% accuracy—failing to outperform the trivial majority-class baseline of 22.20%. Statistical audits utilizing One-Way Analysis of Variance (ANOVA) confirm that the environmental attributes in Dataset 2 exhibit no statistically significant class separability ($p \gg 0.05$ across all features, including soil pH at $p = 0.96$ and temperature at $p = 0.99$). Furthermore, our multi-protocol framework demonstrates that single-split evaluations introduce substantial variance ($\pm 5.94\%$), creating an illusion of model superiority. We conclude that published 99% accuracies in precision crop recommendation are predominantly artifacts of cleanly clustered benchmark distributions rather than algorithmic robustness, and we provide concrete guidelines for trustworthy, reproducible validation in agricultural AI.

**Keywords:** Smart Agriculture, Precision Crop Recommendation, Supervised Learning, Cross-Validation, Split-Selection Bias, Model Collapse, Data Leakage, Empirical Benchmarking.

---

## 1. Introduction

Precision agriculture leverages data-driven intelligence to optimize agricultural inputs, maximize crop yield, and mitigate risks associated with climate change and soil degradation. For smallholder and marginal farmers, selecting the optimal crop suited to localized edaphic (soil nutrients, pH, moisture) and climatic (temperature, humidity, precipitation) factors is one of the most critical decisions governing seasonal economic viability. Over the past five years, the application of classical supervised machine learning algorithms—ranging from tree ensembles to margin-based classifiers—has proliferated across agronomic computing.

A survey of current literature reveals a remarkable consensus: numerous published studies report classification accuracies ranging from 97% to 99.8% for crop recommendation tasks. Such near-perfect performance suggests that the problem of soil- and climate-driven crop selection has been essentially solved by standard algorithms such as Random Forest, XGBoost, and Support Vector Machines. 

However, a rigorous inspection of the experimental methodology underpinning these studies exposes several foundational weaknesses:
1. **Pervasive Reliance on Single Splits:** Most published works report classification metrics derived from a single, arbitrary train-test split (typically 80/20 or 70/30). Without reporting metric dispersion across multiple random seeds, it is impossible to determine whether reported performance reflects average algorithmic efficacy or a fortunate, "lucky" partition of the data.
2. **Benchmark Homogeneity:** The overwhelming majority of crop recommendation studies evaluate their systems exclusively on a single publicly available benchmark dataset (the Kaggle Crop Recommendation dataset containing 2,200 instances). The generalizability of these tuned models to noisier, operational agricultural IoT sensor logs remains unvalidated.
3. **Implicit Data Leakage in Preprocessing:** Many pipelines apply global standardization (e.g., `StandardScaler`) or imputation across the entire dataset prior to partitioning, leaking statistical information from the test set into the training phase.

### Research Questions
This investigation addresses these methodological concerns by formulating three core research questions:
* **RQ1 (Algorithmic Invariance):** Do state-of-the-art classical ML classifiers maintain their high predictive accuracy when evaluated across multiple matched validation protocols?
* **RQ2 (Cross-Dataset Generalization):** Do models tuned on benchmark crop data generalize when deployed on secondary tabular agricultural sensor datasets with real-world noise and missingness?
* **RQ3 (Split-Selection Bias):** How much variance is introduced by arbitrary single holdout splits compared to stratified cross-validation and repeated multi-seed evaluations?

### Major Contributions
To answer these questions, this paper makes the following contributions:
* **Controlled 5×2×3 Benchmarking Harness:** We design and implement an end-to-end evaluation harness comparing 5 algorithm families (Random Forest, XGBoost, SVM-RBF, KNN, Gaussian Naive Bayes) across 2 datasets (clean benchmark vs. noisy sensor logs) and 3 validation protocols (Single Split, 5-Fold Stratified CV, 10× Repeated Stratified Holdout), generating 160 distinct evaluation runs.
* **Rigorous Zero-Leakage Pipeline:** We enforce strict separation of preprocessing transformations, fitting feature scalers exclusively on training splits/folds and freezing all hyperparameter configurations to eliminate data snooping.
* **Empirical Demonstration of Model Collapse:** We prove that all five algorithms collapse to near-random performance (~17%–23%) on secondary agricultural sensor data, failing to exceed the 22.20% majority class baseline.
* **Statistical Feature Auditing via ANOVA:** We provide an empirical explanation for this collapse using One-Way ANOVA hypothesis testing, demonstrating that lack of physical class separability in sensor logs prevents supervised learning.
* **Quantification of the "Lucky Split" Fallacy:** We demonstrate that single unstratified splits exaggerate classification accuracy by up to 5%, establishing the necessity of repeated holdouts and confidence bounds in smart farming research.

---

## 2. Related Work

### 2.1 Machine Learning in Precision Agriculture
The integration of predictive modeling into precision agriculture has spanned several domains, including crop yield prediction, soil nutrient estimation, weed detection, and crop recommendation. Early agricultural decision-support systems relied on rule-based heuristics and linear statistical models. With the advent of modern tabular learning, classical algorithms gained widespread adoption due to their low computational footprint and ease of deployment on edge devices.

Maji et al. (2026) conducted a comparative study of eight classical classifiers and proposed a soft-voting ensemble combining Random Forest, XGBoost, Gradient Boosting, and Gaussian Naive Bayes on the Kaggle Crop Recommendation dataset. The authors reported a test accuracy of 99.55% and an inference latency under 14 ms, highlighting Gaussian Naive Bayes as an exceptionally compact model (3.4 KB) for resource-constrained edge hardware. Similarly, related studies using algorithms like Decision Trees, Support Vector Machines, and Multi-Layer Perceptrons have repeatedly cited accuracies between 96% and 99.2% on the same benchmark.

### 2.2 Methodological Deficiencies in Agricultural AI
Despite stellar reported metrics, the broader machine learning literature has increasingly warned against evaluation artifacts in tabular benchmarking. In deep learning and tabular domains alike, studies have documented the prevalence of "split bias"—the phenomenon where random partition seeds induce large swings in test accuracy, leading researchers to unintentionally report cherry-picked splits.

Furthermore, dataset separability plays a decisive role. Benchmark datasets frequently undergo aggressive curation, outlier trimming, and synthetic augmentation, resulting in hyper-separable clusters where Euclidean or decision-boundary margins are artificially broad. When identical algorithms are exposed to in-situ agricultural IoT data—characterized by sensor calibration drift, missing records, and micro-climatic fluctuations—model performance frequently deteriorates. However, systematic cross-dataset evaluations under matched protocols have been virtually absent in published crop recommendation literature.

---

## 3. Datasets and Exploratory Data Auditing

To rigorously evaluate algorithmic robustness, our investigation utilizes two distinct tabular datasets representing contrasting points on the spectrum of agricultural data quality:

### 3.1 Dataset 1: Kaggle Benchmark (`Crop_recommendation.csv`)
* **Origin & Usage:** Curated by Atharva Ingle (2020) and widely accepted as the standard reference benchmark in precision farming literature.
* **Dimensions:** 2,200 instances and 8 attributes (7 continuous feature variables and 1 categorical target).
* **Target Classes:** 22 distinct crop categories (e.g., rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee).
* **Class Balance:** Perfectly balanced distribution containing exactly 100 samples per crop class (each representing 4.545% of the total dataset).
* **Features:**
  * Nitrogen content in soil (`N`, ratio, range: 0–140)
  * Phosphorus content in soil (`P`, ratio, range: 5–145)
  * Potassium content in soil (`K`, ratio, range: 5–205)
  * Ambient temperature (`temperature`, °C, range: 8.8–43.7)
  * Relative humidity (`humidity`, %, range: 14.3–99.9)
  * Soil pH (`ph`, range: 3.5–9.9)
  * Rainfall (`rainfall`, mm, range: 20.2–298.6)
* **Data Quality:** Zero missing values, zero extreme outliers, and highly distinct feature spaces for each crop type.

### 3.2 Dataset 2: Smart Farming IoT Sensor Dataset (`smart_farming_data.csv`)
* **Origin & Usage:** Sourced from Atharva Soundankar, representing simulated multi-regional smart farming operational logs across 500 farms in India, the USA, and Africa.
* **Dimensions:** 500 instances and 22 attributes comprising environmental telemetry, operational identifiers, and categorical management variables.
* **Target Variable:** `crop_type`, containing 5 primary commercial crops.
* **Class Balance:** Relatively balanced across the 5 crops:
  * Maize: 111 samples (22.20%) — *Majority Class*
  * Soybean: 108 samples (21.60%)
  * Cotton: 107 samples (21.40%)
  * Wheat: 92 samples (18.40%)
  * Rice: 82 samples (16.40%) — *Minority Class*
* **Trivial Majority-Class Baseline:** Predicting the majority class (Maize) for every sample yields a baseline accuracy of **22.20%**.
* **Features Selected for Modeling (9 Attributes):**
  * Continuous Environmental Telemetry: `soil_moisture_%`, `soil_pH`, `temperature_C`, `rainfall_mm`, `humidity_%`, `sunlight_hours`, `NDVI_index`.
  * Categorical Operational Telemetry: `irrigation_type`, `fertilizer_type`.
* **Missing Value Profile:**
  * `irrigation_type`: 150 missing records (**30.00% null rate**).
  * `crop_disease_status`: 130 missing records (**26.00% null rate**).
  * All continuous sensor telemetry columns are fully populated.

---

## 4. Methodology and System Architecture

The overall benchmarking architecture is designed around reproducibility, modular evaluation, and strict leakage prevention. 

```
+-------------------------------------------------------------------------+
|                        RAW DATA INGESTION                               |
|   Dataset 1: Benchmark (2,200 x 8)   |   Dataset 2: Sensor (500 x 22)   |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    ZERO-LEAKAGE PREPROCESSING                           |
|   1. Target Encoding: LabelEncoder fit on target variable               |
|   2. Missing Value Imputation: 'irrigation_type' nulls -> 'missing'     |
|   3. Feature Transformation: StandardScaler fit ONLY on Training Folds  |
|      (SVM, KNN: Scaled | RF, XGBoost, GNB: Raw Features)                |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  FROZEN HYPERPARAMETER ENGINE                           |
|   Tuned once on Dataset 1 Train Partition (configs/frozen_params.json)  |
|   RF (n=200) | XGB (lr=0.2, d=7) | SVM (C=50) | KNN (k=5) | GNB        |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                   MATCHED PROTOCOL EVALUATION                           |
|   [Protocol 1] Single 80/20 Holdout Split (stratify=None, seed=0)       |
|   [Protocol 2] 5-Fold Stratified Cross-Validation (seed=42)             |
|   [Protocol 3] 10x Repeated Stratified Holdout (seeds 0 through 9)      |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                     EMPIRICAL & DIAGNOSTIC AUDIT                        |
|   - Accuracy & Macro-F1 (Mean +/- Std) across 5x2x3 Matrix              |
|   - One-Way ANOVA Signal vs. Noise Hypothesis Tests                     |
|   - Confusion Matrix & Majority-Class Collapse Verification             |
+-------------------------------------------------------------------------+
```

### 4.1 Algorithm Mathematical Formulations

#### 1. Random Forest (RF)
Random Forest is an ensemble of $B$ bootstrap-aggregated decision trees $\{T_b\}_{b=1}^B$. For classification with $K$ classes, each tree predicts class probabilities based on split criteria minimizing Gini impurity:
$$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
The ensemble output is determined by soft-voting majority aggregation:
$$\hat{y} = \arg\max_k \frac{1}{B} \sum_{b=1}^B P_b(y=k \mid \mathbf{x})$$

#### 2. Extreme Gradient Boosting (XGBoost)
XGBoost minimizes a regularized objective function combining multi-class log loss and tree complexity:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$
where the regularization term is defined as:
$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
Second-order Taylor approximations are computed iteratively to optimize split finding across boosting rounds.

#### 3. Support Vector Machine (SVM-RBF)
SVM constructs an optimal hyper-plane maximizing the geometric margin between classes in a transformed Hilbert space. For multi-class classification, a one-versus-one (OvO) voting strategy is utilized with the Radial Basis Function (RBF) kernel:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$
The dual optimization problem subject to box constraints is:
$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \; \sum_{i=1}^n \alpha_i y_i = 0$$

#### 4. K-Nearest Neighbors (KNN)
KNN classifies an unlabelled query instance $\mathbf{x}_q$ by identifying its $k$ nearest neighbors $\mathcal{N}_k(\mathbf{x}_q)$ under the Manhattan ($L_1$) distance metric:
$$d_1(\mathbf{x}_i, \mathbf{x}_j) = \sum_{m=1}^M |x_{im} - x_{jm}|$$
Class probability is weighted inversely by distance:
$$P(y=c \mid \mathbf{x}_q) = \frac{\sum_{i \in \mathcal{N}_k, y_i=c} \frac{1}{d_1(\mathbf{x}_q, \mathbf{x}_i)}}{\sum_{i \in \mathcal{N}_k} \frac{1}{d_1(\mathbf{x}_q, \mathbf{x}_i)}}$$

#### 5. Gaussian Naive Bayes (GNB)
Under the assumption of class-conditional feature independence, GNB applies Bayes' theorem:
$$P(y=c \mid \mathbf{x}) \propto P(y=c) \prod_{m=1}^M \mathcal{N}(x_m \mid \mu_{cm}, \sigma_{cm}^2 + \epsilon)$$
where Gaussian probability density is modeled with variance smoothing parameter $\epsilon = 10^{-9}$ for numerical stability:
$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

### 4.2 Hyperparameter Freezing Protocol
To prevent data snooping and ensure unbiased evaluation across protocols, hyperparameter configurations were established strictly once on the training partition of Dataset 1 (80/20 split, $n=1{,}760$, seed=42) via 5-fold GridSearchCV and saved to `configs/frozen_params.json`:
* **Random Forest:** `n_estimators=200`, `min_samples_split=5`, `max_depth=None`
* **XGBoost:** `n_estimators=200`, `learning_rate=0.2`, `max_depth=7`, `eval_metric='mlogloss'`
* **SVM (RBF):** `C=50.0`, `gamma='scale'`, `kernel='rbf'`
* **KNN:** `n_neighbors=5`, `p=1` (Manhattan), `weights='distance'`
* **Gaussian Naive Bayes:** `var_smoothing=1e-09`

### 4.3 Zero-Leakage Preprocessing Pipeline
Data preprocessing adheres to two strict rules:
1. **Target Encoding:** Target labels (`label` in Dataset 1, `crop_type` in Dataset 2) are mapped to discrete integer labels $[0, K-1]$ using `sklearn.preprocessing.LabelEncoder`.
2. **Model-Specific Feature Scaling:** `StandardScaler` is applied **strictly** to distance- and margin-based models (SVM and KNN). Tree-based ensembles (RF, XGBoost) and Gaussian Naive Bayes receive raw unscaled features. For every fold and split, the scaler is fit **only on training samples** ($X_{\text{train}}$) and applied to evaluate test samples ($X_{\text{test}}$), guaranteeing complete isolation against look-ahead bias.
3. **Missing Value Handling (Dataset 2):** Categorical missing entries in `irrigation_type` are assigned a dedicated sentinel category token (`'missing'`) prior to label encoding, preserving sample volume without synthetic data generation.

---

## 5. Experimental Design and Evaluation Protocols

### 5.1 The Three Matched Evaluation Protocols
To assess the impact of validation methodology on performance metrics, every algorithm is tested under three distinct protocols:

* **Protocol 1: Single Holdout Split ("The Lucky Split Test")**
  * Train/test partition: 80% train ($n=1{,}760$ for D1, $n=400$ for D2), 20% test ($n=440$ for D1, $n=100$ for D2).
  * Partition specification: `train_test_split(..., test_size=0.2, random_state=0, stratify=None)`.
  * Purpose: Evaluates vulnerability to non-stratified sampling artifacts commonly found in literature.
* **Protocol 2: 5-Fold Stratified Cross-Validation**
  * Partition specification: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
  * Purpose: Measures out-of-fold generalizability across 100% of dataset instances while enforcing balanced class representations.
* **Protocol 3: 10× Repeated Stratified Holdout**
  * Partition specification: 10 independent stratified 80/20 splits generated across pseudo-random seeds $s \in \{0, 1, 2, \dots, 9\}$.
  * Purpose: Quantifies split-selection variance ($\pm \sigma$) and defines robust empirical confidence bounds.

### 5.2 Performance Metrics
Model performance is evaluated across two primary metrics:
* **Overall Classification Accuracy:**
  $$\text{Accuracy} = \frac{\sum_{k=1}^K TP_k}{N}$$
* **Macro-Averaged F1-Score:**
  $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
Macro-F1 assigns equal weight to every class regardless of sample volume, serving as the definitive measure of class-wise robustness under class imbalance.

---

## 6. Empirical Results and Comparative Analysis

### 6.1 The Complete 5×2×3 Experimental Grid
Table 1 presents the complete performance results across all 30 experimental conditions, reporting mean accuracy and macro-F1 alongside standard deviations across folds and seeds.

#### Table 1: Master Empirical Evaluation Matrix (Accuracy and Macro-F1 across 5 Models, 2 Datasets, and 3 Protocols)
| Dataset | Algorithm Family | Protocol 1: Single Split (Acc / F1) | Protocol 2: 5-Fold CV (Acc ± $\sigma$ / F1 ± $\sigma$) | Protocol 3: 10× Repeated (Acc ± $\sigma$ / F1 ± $\sigma$) |
| :--- | :--- | :---: | :---: | :---: |
| **Dataset 1**<br>*(Benchmark)* | **Random Forest** | 0.9977 / 0.9980 | 0.9955 ± 0.0032 / 0.9954 ± 0.0032 | 0.9952 ± 0.0013 / 0.9952 ± 0.0013 |
| | **XGBoost** | 0.9977 / 0.9977 | 0.9941 ± 0.0044 / 0.9941 ± 0.0044 | 0.9889 ± 0.0039 / 0.9888 ± 0.0040 |
| | **SVM (RBF)** | 0.9932 / 0.9940 | 0.9859 ± 0.0044 / 0.9858 ± 0.0045 | 0.9898 ± 0.0031 / 0.9898 ± 0.0031 |
| | **KNN** | 0.9841 / 0.9855 | 0.9809 ± 0.0107 / 0.9807 ± 0.0109 | 0.9823 ± 0.0090 / 0.9822 ± 0.0091 |
| | **Gaussian NB** | 0.9932 / 0.9941 | 0.9945 ± 0.0020 | 0.9957 ± 0.0029 / 0.9957 ± 0.0029 |
| **Dataset 2**<br>*(Noisy Sensor)* | **Random Forest** | **0.2700** / 0.2480 | 0.2260 ± 0.0594 / 0.2062 ± 0.0522 | 0.2330 ± 0.0356 / 0.2090 ± 0.0374 |
| | **XGBoost** | 0.2100 / 0.1909 | 0.2240 ± 0.0351 / 0.2200 ± 0.0349 | 0.1920 ± 0.0297 / 0.1816 ± 0.0285 |
| | **SVM (RBF)** | 0.1700 / 0.1519 | 0.2200 ± 0.0561 / 0.2114 ± 0.0490 | 0.2080 ± 0.0290 / 0.1973 ± 0.0264 |
| | **KNN** | 0.1700 / 0.1703 | 0.2120 ± 0.0239 / 0.2071 ± 0.0147 | 0.2100 ± 0.0333 / 0.2014 ± 0.0304 |
| | **Gaussian NB** | 0.2200 / 0.1872 | 0.2000 ± 0.0235 / 0.1787 ± 0.0251 | 0.1790 ± 0.0251 / 0.1578 ± 0.0216 |

*(Dataset 1 Majority Baseline = **4.55%**; Dataset 2 Majority Baseline = **22.20%**)*

```
[Insert Figure 1: results/figures/fig1_cross_dataset_comparison.png]
Figure 1: Cross-Dataset Generalization Breakdown across 5 ML algorithms under 5-Fold Cross-Validation, illustrating the precipitous drop from >98% accuracy on benchmark data to baseline levels on operational sensor data.
```

### 6.2 Analysis of Benchmark Performance (Dataset 1)
On Dataset 1, all five algorithms validate the findings of prior literature. Random Forest achieves a peak 5-fold CV accuracy of **99.55% ± 0.32%**, followed closely by Gaussian Naive Bayes at **99.45% ± 0.20%** and XGBoost at **99.41% ± 0.44%**. SVM and KNN deliver **98.59%** and **98.09%**, respectively. Metric dispersion across the 10 repeated holdouts is negligible ($\sigma \leq 0.0039$ for all models except KNN at $\sigma = 0.0090$), proving that on cleanly clustered benchmark data, classification performance is highly stable and virtually independent of partition seed selection.

### 6.3 Catastrophic Collapse on Operational Sensor Data (Dataset 2)
When exposed to Dataset 2, the exact same model instances experience catastrophic performance failure. Under 5-fold cross-validation:
* Random Forest drops from 99.55% to **22.60% ± 5.94%**.
* XGBoost drops from 99.41% to **22.40% ± 3.51%**.
* SVM drops from 98.59% to **22.00% ± 5.61%**.
* KNN drops from 98.09% to **21.20% ± 2.39%**.
* Gaussian Naive Bayes drops from 99.45% to **20.00% ± 2.35%**.

Critically, the majority-class baseline for Dataset 2 (Maize, 111 out of 500 instances) is **22.20%**. A completely uninformative dummy classifier that predicts Maize uniformly achieves 22.20% accuracy. **Not a single supervised machine learning model demonstrates statistically meaningful predictive capability beyond trivial chance.**

---

## 7. Collapse Mode Diagnosis and Feature Separability Audit

To uncover the root cause of this performance collapse, we conducted two diagnostic investigations: a One-Way ANOVA feature separability audit and an out-of-fold confusion matrix error analysis.

### 7.1 Statistical Feature Separability Audit via One-Way ANOVA
We performed One-Way Analysis of Variance (ANOVA) tests on all continuous environmental attributes across the five crop categories in Dataset 2. Under the null hypothesis $H_0$, the true population means of a given environmental attribute are identical across all crop classes:
$$\mu_{\text{Cotton}} = \mu_{\text{Maize}} = \mu_{\text{Rice}} = \mu_{\text{Soybean}} = \mu_{\text{Wheat}}$$
A statistically significant agricultural signal requires rejecting $H_0$ ($p < 0.05$), demonstrating that crops occupy differentiated environmental niches.

#### Table 2: One-Way ANOVA Test for Feature Separability Across Crops in Dataset 2
| Feature Attribute | Cotton (Mean ± $\sigma$) | Maize (Mean ± $\sigma$) | Rice (Mean ± $\sigma$) | Soybean (Mean ± $\sigma$) | Wheat (Mean ± $\sigma$) | $F$-Statistic | $p$-value | Agronomic Separability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`soil_pH`** | 6.54 ± 0.57 | 6.52 ± 0.62 | 6.49 ± 0.58 | 6.52 ± 0.60 | 6.55 ± 0.55 | 0.1536 | **0.9614** | **No Separability (Noise)** |
| **`temperature_C`** | 24.63 ± 5.72 | 24.63 ± 5.31 | 24.63 ± 5.63 | 24.66 ± 5.04 | 24.84 ± 5.16 | 0.0258 | **0.9987** | **No Separability (Noise)** |
| **`rainfall_mm`** | 190.1 ± 63.4 | 177.0 ± 78.1 | 173.6 ± 76.1 | 187.4 ± 70.3 | 178.0 ± 73.7 | 0.9589 | **0.4297** | **No Separability (Noise)** |
| **`soil_moisture_%`** | 26.53 ± 10.22 | 26.58 ± 10.06 | 27.44 ± 10.64 | 28.34 ± 10.37 | 24.73 ± 9.27 | 1.6962 | **0.1496** | **No Separability (Noise)** |
| **`humidity_%`** | 63.81 ± 15.76 | 64.87 ± 14.16 | 64.02 ± 14.96 | 67.44 ± 14.29 | 65.61 ± 13.96 | 1.0405 | **0.3857** | **No Separability (Noise)** |
| **`sunlight_hours`** | 7.16 ± 1.64 | 6.97 ± 1.74 | 7.17 ± 1.56 | 7.05 ± 1.75 | 6.81 ± 1.74 | 0.7317 | **0.5706** | **No Separability (Noise)** |

```
[Insert Figure 3: results/figures/fig3_anova_feature_distributions.png]
Figure 2: Distribution boxplots of environmental telemetry across crop categories in Dataset 2, demonstrating uniform distributions and lack of class separation.
```

The statistical audit results in Table 2 are definitive: **every environmental feature yields $p \gg 0.05$**. In particular, ambient temperature ($p = 0.9987$) and soil pH ($p = 0.9614$) exhibit identical distribution parameters across all five crop categories. In reality, crops such as rice require high water and specific soil conditions compared to cotton or wheat. In Dataset 2, however, sensor values were generated independently of crop assignment. Because the underlying data lacks physical class separability, the machine learning algorithms are attempting to learn non-existent patterns from uniform noise.

### 7.2 Out-of-Fold Error Analysis and Confusion Matrix Diagnosis
To examine how models fail, we computed out-of-fold confusion matrices for Random Forest and XGBoost across the 5-fold cross-validation procedure.

```
[Insert Figure 4: results/figures/fig4_confusion_matrices_dataset2.png]
Figure 3: Out-of-fold confusion matrix heatmaps for Random Forest and XGBoost on Dataset 2, illustrating uniform prediction dispersion characteristic of representation collapse.
```

#### Table 3: Per-Class F1-Score Matrix on Dataset 2 (Out-of-Fold 5-Fold CV)
| Crop Class | Support ($N$) | Random Forest ($F_1$) | XGBoost ($F_1$) | SVM-RBF ($F_1$) | KNN ($F_1$) | Gaussian NB ($F_1$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cotton** | 107 | 0.2712 | 0.2315 | 0.2896 | 0.2559 | 0.2241 |
| **Maize** *(Majority)* | 111 | 0.2302 | 0.2203 | 0.2551 | 0.2058 | 0.2092 |
| **Rice** *(Minority)* | 82 | 0.1260 | 0.1892 | 0.1899 | 0.1974 | 0.0364 |
| **Soybean** | 108 | 0.3172 | 0.2698 | 0.1980 | 0.2407 | 0.2646 |
| **Wheat** | 92 | 0.1013 | 0.1946 | 0.1364 | 0.1461 | 0.1605 |
| **Macro-Average F1** | 500 | **0.2062** | **0.2200** | **0.2114** | **0.2071** | **0.1787** |

Table 3 reveals that the models do not collapse to a single majority class (which would yield $F_1 \approx 0$ on all minority classes). Instead, they exhibit **stochastic representation collapse**: the algorithms scatter predictions uniformly across all five classes, achieving an empirical accuracy of $\approx 1/5 = 20\%$. Minority crops suffer disproportionately: Gaussian Naive Bayes achieves an $F_1$ of only **0.0364** on Rice, and Random Forest scores **0.1013** on Wheat.

### 7.3 Quantifying the "Lucky Split" Fallacy
A central finding of our work is the quantification of split-selection bias. On Dataset 2, Random Forest achieved an accuracy of **27.00%** under the unstratified Single Split protocol (Table 1). A researcher reporting only this single split would claim a $+4.80\%$ improvement over the majority-class baseline. 

```
[Insert Figure 2: results/figures/fig2_split_variance_bias.png]
Figure 4: Single Split vs. 10x Repeated Holdout on Dataset 2, showing how arbitrary single partitions overestimate performance.
```

However, under 5-fold cross-validation, Random Forest’s true performance is **22.60% ± 5.94%**, and across 10 repeated holdouts, it is **23.30% ± 3.56%**. The single-split score was simply an outlier caused by favorable random partitioning. In noisy or poorly separated domains, arbitrary single splits introduce substantial metric variance, creating a false impression of predictive competence.

---

## 8. Discussion and Agronomic Implications

### 8.1 The Ceiling Accuracy Paradox in Smart Agriculture
Our findings provide a critical reinterpretation of published crop recommendation literature:
* **The 99% accuracy reported in dozens of recent papers is not an indicator of algorithm superiority.** Rather, it is an artifact of the Kaggle Crop Recommendation dataset, whose samples form tightly clustered, well-isolated hyper-volumes in 7-dimensional space. Under such conditions, even elementary classifiers like KNN and Naive Bayes easily separate the classes.
* **When classical ML models are deployed on operational sensor telemetry, high performance cannot be assumed.** If IoT sensor streams lack physical agronomic correlation with crop suitability, even complex gradient-boosted ensembles collapse to random chance.

### 8.2 Practical Recommendations for Trustworthy Agricultural AI
To prevent misleading claims in future precision agriculture research, we propose four essential methodological guidelines:
1. **Mandatory Multi-Seed Protocol Reporting:** Authors must report mean performance and standard deviations across at least 5-fold cross-validation or 10× repeated holdout splits. Single-split results must not be accepted as evidence of model capability.
2. **Pre-Training Signal Auditing:** Before training complex models, researchers must perform baseline statistical tests (such as ANOVA, mutual information, or Kruskal-Wallis tests) to verify that features exhibit statistically significant class separability.
3. **Explicit Reporting Against Majority Baselines:** In multi-class classification, accuracy numbers must always be benchmarked against the zero-rule majority-class baseline ($\text{Acc}_{\text{maj}} = \max_k N_k / N$).
4. **Strict Isolation of Data Preprocessing:** All transformations—including scalers, normalizers, and categorical encoders—must be fit exclusively on training partitions to prevent subtle data leakage.

---

## 9. Conclusion and Future Work

This paper conducted a rigorous empirical audit of five supervised machine learning algorithm families across two agricultural datasets and three matched evaluation protocols. Our findings demonstrate that:
1. The near-ceiling classification accuracies (98%–99.8%) reported in smart-agriculture literature reflect the clean separability of benchmark data rather than algorithmic robustness.
2. On secondary agricultural sensor logs, all five algorithms experienced complete representation collapse (~17%–23%), failing to beat the 22.20% majority-class baseline due to lack of physical feature separability ($p \gg 0.05$ across all ANOVA tests).
3. Single unstratified splits introduce severe variance ($\pm 5.94\%$), confirming that single-split reporting in precision agriculture can create a deceptive illusion of performance.

**Future Work:** We plan to expand this benchmark by incorporating multi-modal satellite remote sensing telemetry (e.g., Sentinel-2 NDVI time-series) and soil spectral signatures, and to explore conformal prediction frameworks that produce reliable prediction sets with rigorous statistical error bounds under real-world agricultural uncertainty.

---

## References

1. Maji, C., Pal, P., Singh, R. K., & Upadhyay, S. (2026). Soil and Climate-Driven Crop Recommendation via Ensemble Learning: A Comparative Study of Classical Classifiers for Resource-Constrained Deployment. *Proceedings of the International Conference on Intelligent Systems and Robotics for Sustainable Development (ISRSD-2026)*, Springer Lecture Notes in Electrical Engineering (LNEE).
2. Zanzari, K., et al. (2026). A Comparative Benchmarking Study of Classical Machine Learning and Deep Learning Methods for Image-Based Deepfake Detection. *Applied Cybersecurity & Internet Governance (ACIG)*, 5(1), DOI: 10.60097/ACIG/221087.
3. Ingle, A. (2020). Crop Recommendation Dataset. *Kaggle Datasets*. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset.
4. Soundankar, A. (2024). Smart Farming Sensor Data for Yield Prediction. *Kaggle Datasets*. https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction.
5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
7. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273-297.
8. Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
10. Kapoor, S., & Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. *Patterns*, 4(9), 100804.
11. Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., ... & Varoquaux, G. (2021). Accounting for Variance in Machine Learning Benchmarks. *Proceedings of Machine Learning and Systems (MLSys)*, 3, 253-269.
12. Varoquaux, G. (2018). Cross-Validation Failure: Small Sample Sizes Lead to Large Error Bars. *NeuroImage*, 180, 68-77.
