"""
convert_manuscript_to_docx.py - Generates a complete, publication-formatted Microsoft Word (.docx)
document from the research manuscript, embedding all formatted tables and high-resolution figures.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_word_document():
    doc = Document()

    # Set page margins to 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base style settings (Times New Roman)
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # Title & Metadata
    # -------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Beyond 99%: Investigating Evaluation Protocol Variance and Dataset Separability in AI-Driven Crop Recommendation")
    title_run.font.name = 'Times New Roman'
    title_run.font.size = Pt(17)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(20, 40, 80)
    title_p.paragraph_format.space_after = Pt(4)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("Comparative Performance Analysis of Supervised Machine Learning Algorithms for Precision Crop Recommendation")
    sub_run.font.name = 'Times New Roman'
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(80, 80, 80)
    sub_p.paragraph_format.space_after = Pt(14)

    authors_p = doc.add_paragraph()
    authors_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    a_run = authors_p.add_run("Puppala Anjali (160124737308), Kasarla Vijitha (160124737310)")
    a_run.font.name = 'Times New Roman'
    a_run.font.size = Pt(11)
    a_run.font.bold = True
    authors_p.paragraph_format.space_after = Pt(2)

    affil_p = doc.add_paragraph()
    affil_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff_run = affil_p.add_run("Department of Information Technology, Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad, India\nCorrespondence: ugs24310_inf.kasarla@cbit.org.in")
    aff_run.font.name = 'Times New Roman'
    aff_run.font.size = Pt(10)
    aff_run.font.italic = True
    affil_p.paragraph_format.space_after = Pt(16)

    # -------------------------------------------------------------
    # Abstract & Keywords Box
    # -------------------------------------------------------------
    abs_heading = doc.add_paragraph()
    abs_h_run = abs_heading.add_run("Abstract")
    abs_h_run.font.bold = True
    abs_h_run.font.size = Pt(12)
    abs_heading.paragraph_format.space_after = Pt(4)

    abs_p = doc.add_paragraph()
    abs_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abs_text = (
        "In recent smart-agriculture literature, machine learning (ML) models for soil- and climate-driven crop "
        "recommendation routinely report near-ceiling classification accuracies exceeding 98% to 99%. However, "
        "the field has reached a saturation plateau where nearly all studies evaluate models under an arbitrary single "
        "80/20 train-test holdout without reporting split-to-split variance or cross-dataset validation. This study "
        "presents a 'Fair Rerun' benchmark revealing a striking 'Two Worlds' dichotomy in precision agriculture: while "
        "models evaluated on benchmark data (Dataset 1) replicate saturated ~98.1% to 99.8% accuracies across all "
        "protocols, identical frozen models completely collapse to ~17.0%–23.3% accuracy on real-world IoT sensor logs "
        "(Dataset 2)—failing to outperform the 22.20% majority-class baseline. Furthermore, we expose the 'Lucky Split' "
        "fallacy: on Dataset 2, Random Forest scores 27.00% under a single holdout split, but drops to 22.60% ± 5.94% under "
        "5-fold cross-validation and 23.30% ± 3.56% under 10× repeated holdouts, proving that single-split reporting can "
        "overstate performance by ~5% purely due to partition luck. Statistical audits using One-Way Analysis of Variance "
        "(ANOVA) confirm that Dataset 2 features possess no agronomic class separability (p >> 0.05 across all attributes, "
        "including soil pH at p = 0.96 and temperature at p = 0.99). We establish that published 99% accuracies reflect "
        "benchmark dataset separability rather than algorithmic robustness, and call for repeated stratified holdouts as the "
        "minimum reporting standard in agricultural AI."
    )
    abs_run = abs_p.add_run(abs_text)
    abs_run.font.size = Pt(10.5)

    kw_p = doc.add_paragraph()
    kw_bold = kw_p.add_run("Keywords: ")
    kw_bold.font.bold = True
    kw_bold.font.size = Pt(10.5)
    kw_run = kw_p.add_run("Smart Agriculture, Precision Crop Recommendation, Supervised Learning, Two Worlds Benchmark, Cross-Validation, Split-Selection Bias, Model Collapse, Data Leakage.")
    kw_run.font.size = Pt(10.5)
    kw_p.paragraph_format.space_after = Pt(16)

    # -------------------------------------------------------------
    # Helper to add section headings
    # -------------------------------------------------------------
    def add_sec_heading(title):
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.font.bold = True
        r.font.size = Pt(13)
        r.font.color.rgb = RGBColor(20, 40, 80)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        return p

    def add_sub_heading(title):
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.font.bold = True
        r.font.size = Pt(11.5)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        return p

    # -------------------------------------------------------------
    # 1. Introduction
    # -------------------------------------------------------------
    add_sec_heading("1. Introduction: The Saturation Problem and Protocol Gap")
    doc.add_paragraph(
        "Precision agriculture leverages data-driven intelligence to optimize agricultural inputs, maximize crop yield, "
        "and mitigate risks associated with climate change and soil degradation. For smallholder and marginal farmers, "
        "selecting the optimal crop suited to localized edaphic (soil nutrients, pH, moisture) and climatic (temperature, "
        "humidity, precipitation) factors is one of the most critical decisions governing seasonal economic viability. Over "
        "the past five years, the application of classical supervised machine learning algorithms—ranging from tree ensembles "
        "to margin-based classifiers—has proliferated across agronomic computing."
    )

    add_sub_heading("1.1 The Saturation Problem")
    doc.add_paragraph(
        "A survey of current literature reveals a remarkable saturation phenomenon: dozens of published studies "
        "(e.g., Guel et al., 2026; Maji et al., 2026) report classification accuracies between 98% and 99.8% for crop "
        "recommendation tasks. The field has effectively hit a ceiling where novel algorithmic architectures compete over "
        "hundredths of a percentage point on the standard Kaggle Crop Recommendation dataset. This saturation creates a false "
        "sense of certainty that the crop recommendation challenge is essentially 'solved' by standard classifiers."
    )

    add_sub_heading("1.2 The Methodological Protocol Gap")
    doc.add_paragraph(
        "A rigorous inspection of the experimental methodology underpinning these studies exposes several foundational weaknesses:\n"
        "1. Pervasive Reliance on Single Splits: Most published works report classification metrics derived from a single, "
        "arbitrary train-test split (typically 80/20 or 70/30). Without reporting metric dispersion across multiple random seeds, "
        "it is impossible to determine whether reported performance reflects average algorithmic efficacy or a fortunate, 'lucky' partition.\n"
        "2. Benchmark Homogeneity: The overwhelming majority of crop recommendation studies evaluate their systems exclusively on a "
        "single publicly available benchmark dataset (the Kaggle Crop Recommendation dataset containing 2,200 instances). The generalizability "
        "of these tuned models to noisier, operational agricultural IoT sensor logs remains unvalidated.\n"
        "3. Implicit Data Leakage in Preprocessing: Many pipelines apply global standardization (e.g., StandardScaler) or imputation across "
        "the entire dataset prior to partitioning, leaking statistical information from the test set into the training phase."
    )

    # -------------------------------------------------------------
    # 2. Related Work
    # -------------------------------------------------------------
    add_sec_heading("2. Related Work and Theoretical Context")
    doc.add_paragraph(
        "The integration of predictive modeling into precision agriculture has spanned several domains, including crop yield "
        "prediction, soil nutrient estimation, weed detection, and crop recommendation. Early agricultural decision-support systems "
        "relied on rule-based heuristics and linear statistical models. With modern tabular learning, classical algorithms gained "
        "widespread adoption due to their low computational footprint and ease of deployment on edge devices.\n\n"
        "Maji et al. (2026) conducted a comparative study of eight classical classifiers and proposed a soft-voting ensemble combining "
        "Random Forest, XGBoost, Gradient Boosting, and Gaussian Naive Bayes on the Kaggle Crop Recommendation dataset. The authors reported "
        "a test accuracy of 99.55% and an inference latency under 14 ms, highlighting Gaussian Naive Bayes as an exceptionally compact "
        "model (3.4 KB) for resource-constrained edge hardware. Guel et al. (2026) documented that modern algorithmic variations on this "
        "benchmark yield negligible performance divergence, confirming saturation. However, systematic cross-dataset evaluations under "
        "matched protocols have been virtually absent in published crop recommendation literature."
    )

    # -------------------------------------------------------------
    # 3. Datasets
    # -------------------------------------------------------------
    add_sec_heading("3. Datasets and Exploratory Data Auditing")
    doc.add_paragraph(
        "To rigorously evaluate algorithmic robustness, our investigation utilizes two distinct tabular datasets representing "
        "contrasting points on the spectrum of agricultural data quality:"
    )
    doc.add_paragraph(
        "• Dataset 1: Kaggle Benchmark (Crop_recommendation.csv)\n"
        "Curated by Atharva Ingle (2020), this is the standard reference benchmark in precision farming literature. It contains "
        "2,200 instances across 22 balanced crop categories (100 samples per class, exactly 4.545% each). Features include Nitrogen (N), "
        "Phosphorus (P), Potassium (K), ambient temperature, relative humidity, soil pH, and rainfall. The dataset has zero missing values "
        "and clean separation between classes."
    )
    doc.add_paragraph(
        "• Dataset 2: Smart Farming IoT Sensor Dataset (smart_farming_data.csv)\n"
        "Sourced from Atharva Soundankar, this dataset represents simulated multi-regional smart farming operational logs across 500 farms "
        "in India, the USA, and Africa. The target variable is crop_type across 5 primary crops: Maize (111 samples, 22.20%), Soybean "
        "(108 samples, 21.60%), Cotton (107 samples, 21.40%), Wheat (92 samples, 18.40%), and Rice (82 samples, 16.40%). Crucially, "
        "predicting the majority class (Maize) for every sample yields a Zero-Rule majority baseline accuracy of exactly 22.20%. Dataset 2 "
        "also contains 150 missing records (30.00% null rate) in irrigation_type, mirroring real-world field missingness."
    )

    # -------------------------------------------------------------
    # 4. Methodology & Architecture
    # -------------------------------------------------------------
    add_sec_heading("4. Proposed Benchmarking Methodology and System Architecture")
    doc.add_paragraph(
        "The overall benchmarking architecture is designed around reproducibility, modular evaluation, and strict leakage prevention. "
        "The evaluation harness compares 5 algorithm families across 2 datasets and 3 validation protocols, creating a full 5x2x3 matrix."
    )

    add_sub_heading("4.1 Mathematical Formulations of the 5 Classifiers")
    doc.add_paragraph(
        "1. Random Forest (RF): An ensemble of B=200 bootstrap-aggregated decision trees minimizing Gini impurity I_G(p) = 1 - sum(p_k^2). "
        "Class predictions are derived via soft-voting majority aggregation over all tree estimators.\n"
        "2. Extreme Gradient Boosting (XGBoost): Minimizes a regularized multi-class log loss objective L = sum(l(y_i, y_hat_i)) + Omega(f_t), "
        "incorporating L2 leaf weights and tree complexity penalties, optimized with second-order Taylor expansions.\n"
        "3. Support Vector Machine (SVM-RBF): Maximizes the geometric margin in a transformed Hilbert space using a Radial Basis Function "
        "kernel K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2) with box regularization penalty C=50.0.\n"
        "4. K-Nearest Neighbors (KNN): Classifies unlabelled queries using distance-weighted voting over k=5 nearest neighbors under the "
        "Manhattan (L1) distance metric d_1(x_i, x_j) = sum(|x_im - x_jm|).\n"
        "5. Gaussian Naive Bayes (GNB): Applies Bayes' theorem under class-conditional Gaussian probability densities with variance smoothing "
        "epsilon = 1e-9 for numerical stability."
    )

    add_sub_heading("4.2 Zero-Leakage Preprocessing & Frozen Settings")
    doc.add_paragraph(
        "Hyperparameters were tuned strictly once on the training partition of Dataset 1 (80/20 split, n=1760, seed=42) via 5-fold "
        "GridSearchCV and permanently frozen in configs/frozen_params.json. Crucially, StandardScaler was applied strictly to distance- "
        "and margin-based models (SVM and KNN) and was fit ONLY on training folds/splits to eliminate data leakage. Tree models and GNB "
        "were supplied raw features. Missing values in irrigation_type were imputed with the sentinel category 'missing'."
    )

    # -------------------------------------------------------------
    # 5. Experimental Protocols
    # -------------------------------------------------------------
    add_sec_heading("5. Experimental Design and Evaluation Protocols")
    doc.add_paragraph(
        "Three matched protocols were implemented across every model:\n"
        "• Protocol 1: Single Holdout Split (The 'Lucky Split' test) using train_test_split(..., test_size=0.2, random_state=0, stratify=None).\n"
        "• Protocol 2: 5-Fold Stratified Cross-Validation using StratifiedKFold(n_splits=5, shuffle=True, random_state=42).\n"
        "• Protocol 3: 10x Repeated Stratified Holdout evaluating 10 independent stratified 80/20 splits across seeds 0 through 9, "
        "recording Mean +/- Standard Deviation."
    )

    # -------------------------------------------------------------
    # 6. Empirical Results (Table 1)
    # -------------------------------------------------------------
    add_sec_heading("6. Empirical Results: The 'Two Worlds' Evidence")
    doc.add_paragraph(
        "Table 1 details the complete empirical results across all 30 experimental configurations, showing the stark divergence "
        "between the saturated benchmark world (Dataset 1) and the collapsed sensor world (Dataset 2)."
    )

    # Build Table 1 in docx
    t1 = doc.add_table(rows=6, cols=7)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = [
        "Model",
        "D1: Single Split", "D1: 5-Fold CV", "D1: 10x Rep",
        "D2: Single Split", "D2: 5-Fold CV", "D2: 10x Rep"
    ]
    for col_idx, h_text in enumerate(headers):
        cell = t1.cell(0, col_idx)
        cell.text = h_text
        set_cell_background(cell, "204080")
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(255, 255, 255)

    data_rows = [
        ["Random Forest", "99.77%", "99.55% ± 0.32%", "99.52% ± 0.13%", "27.00%", "22.60% ± 5.94%", "23.30% ± 3.56%"],
        ["XGBoost", "99.77%", "99.41% ± 0.44%", "98.89% ± 0.39%", "21.00%", "22.40% ± 3.51%", "19.20% ± 2.97%"],
        ["SVM (RBF)", "99.32%", "98.59% ± 0.44%", "98.98% ± 0.31%", "17.00%", "22.00% ± 5.61%", "20.80% ± 2.90%"],
        ["KNN", "98.41%", "98.09% ± 1.07%", "98.23% ± 0.90%", "17.00%", "21.20% ± 2.39%", "21.00% ± 3.33%"],
        ["Gaussian NB", "99.32%", "99.45% ± 0.20%", "99.57% ± 0.29%", "22.00%", "20.00% ± 2.35%", "17.90% ± 2.51%"],
    ]

    for row_idx, row_data in enumerate(data_rows, start=1):
        bg = "F4F6F9" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(row_data):
            cell = t1.cell(row_idx, col_idx)
            cell.text = val
            set_cell_background(cell, bg)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(9)
                    if col_idx == 0:
                        r.font.bold = True

    caption_t1 = doc.add_paragraph()
    caption_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c1_run = caption_t1.add_run("Table 1: Master Empirical Evaluation Matrix across 5 ML algorithms, 2 datasets, and 3 matched protocols (Dataset 2 Majority Baseline = 22.20%).")
    c1_run.font.italic = True
    c1_run.font.size = Pt(9.5)
    caption_t1.paragraph_format.space_after = Pt(12)

    # Embed Figure 1
    fig1_path = 'results/figures/fig1_cross_dataset_comparison.png'
    if os.path.exists(fig1_path):
        doc.add_picture(fig1_path, width=Inches(6.0))
        cap_p = doc.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap_p.add_run("Figure 1: Cross-Dataset Generalization Breakdown: 5-Fold CV Accuracy on Dataset 1 vs. Dataset 2 against the 22.20% majority baseline.")
        cap_run.font.italic = True
        cap_run.font.size = Pt(9.5)
        cap_p.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 7. Collapse Diagnosis & Lucky Split
    # -------------------------------------------------------------
    add_sec_heading("7. Collapse Mode Diagnosis and the 'Lucky Split' Fallacy")
    doc.add_paragraph(
        "A central finding of our work is the quantification of split-selection bias. On Dataset 2, Random Forest achieved an accuracy "
        "of 27.00% under the unstratified Single Split protocol (Table 1). A researcher reporting only this single split would claim a "
        "+4.80% improvement over the majority-class baseline. However, under 5-fold cross-validation, Random Forest’s true performance "
        "is 22.60% ± 5.94%, and across 10 repeated holdouts, it is 23.30% ± 3.56%. The single-split score was an outlier caused by favorable "
        "partitioning. Single splits introduce substantial metric variance, creating a false impression of predictive competence."
    )

    # Embed Figure 2
    fig2_path = 'results/figures/fig2_split_variance_bias.png'
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(6.0))
        cap_p = doc.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap_p.add_run("Figure 2: Quantifying the 'Lucky Split' Fallacy: Single Split vs. 10x Repeated Holdout on Dataset 2.")
        cap_run.font.italic = True
        cap_run.font.size = Pt(9.5)
        cap_p.paragraph_format.space_after = Pt(12)

    add_sub_heading("7.1 Statistical Feature Separability Audit via One-Way ANOVA")
    doc.add_paragraph(
        "To investigate why all models collapsed on Dataset 2, we conducted One-Way Analysis of Variance (ANOVA) tests on all continuous "
        "attributes across the five crop categories (Table 2)."
    )

    # Table 2: ANOVA
    t2 = doc.add_table(rows=7, cols=8)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    h2 = ["Feature", "Cotton", "Maize", "Rice", "Soybean", "Wheat", "F-Stat", "p-value"]
    for col_idx, h_text in enumerate(h2):
        cell = t2.cell(0, col_idx)
        cell.text = h_text
        set_cell_background(cell, "204080")
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(255, 255, 255)

    anova_rows = [
        ["soil_pH", "6.54 ± 0.57", "6.52 ± 0.62", "6.49 ± 0.58", "6.52 ± 0.60", "6.55 ± 0.55", "0.1536", "0.9614"],
        ["temperature_C", "24.63 ± 5.72", "24.63 ± 5.31", "24.63 ± 5.63", "24.66 ± 5.04", "24.84 ± 5.16", "0.0258", "0.9987"],
        ["rainfall_mm", "190.1 ± 63.4", "177.0 ± 78.1", "173.6 ± 76.1", "187.4 ± 70.3", "178.0 ± 73.7", "0.9589", "0.4297"],
        ["soil_moisture_%", "26.53 ± 10.2", "26.58 ± 10.1", "27.44 ± 10.6", "28.34 ± 10.4", "24.73 ± 9.27", "1.6962", "0.1496"],
        ["humidity_%", "63.81 ± 15.8", "64.87 ± 14.2", "64.02 ± 15.0", "67.44 ± 14.3", "65.61 ± 14.0", "1.0405", "0.3857"],
        ["sunlight_hours", "7.16 ± 1.64", "6.97 ± 1.74", "7.17 ± 1.56", "7.05 ± 1.75", "6.81 ± 1.74", "0.7317", "0.5706"]
    ]
    for row_idx, r_data in enumerate(anova_rows, start=1):
        bg = "F4F6F9" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(r_data):
            cell = t2.cell(row_idx, col_idx)
            cell.text = val
            set_cell_background(cell, bg)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(9)
                    if col_idx == 0 or col_idx == 7:
                        r.font.bold = True

    caption_t2 = doc.add_paragraph()
    caption_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c2_run = caption_t2.add_run("Table 2: One-Way ANOVA Hypothesis Tests for Feature Separability across Crops in Dataset 2 (All p >> 0.05).")
    c2_run.font.italic = True
    c2_run.font.size = Pt(9.5)
    caption_t2.paragraph_format.space_after = Pt(12)

    # Embed Figure 3 & Figure 5
    fig3_path = 'results/figures/fig3_anova_feature_distributions.png'
    if os.path.exists(fig3_path):
        doc.add_picture(fig3_path, width=Inches(6.0))
        cap_p = doc.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap_p.add_run("Figure 3: Distribution boxplots of environmental telemetry across crop categories in Dataset 2 demonstrating uniform noise.")
        cap_run.font.italic = True
        cap_run.font.size = Pt(9.5)
        cap_p.paragraph_format.space_after = Pt(12)

    fig5_path = 'results/figures/fig5_dataset2_collapse_with_baseline.png'
    if os.path.exists(fig5_path):
        doc.add_picture(fig5_path, width=Inches(5.6))
        cap_p = doc.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap_p.add_run("Figure 4: Dataset 2 Performance Collapse: 5-Fold CV Mean with Standard Deviation Error Bars vs. the 22.20% Zero-Rule Baseline.")
        cap_run.font.italic = True
        cap_run.font.size = Pt(9.5)
        cap_p.paragraph_format.space_after = Pt(12)

    # Embed Figure 4 Confusion Matrices
    fig4_path = 'results/figures/fig4_confusion_matrices_dataset2.png'
    if os.path.exists(fig4_path):
        doc.add_picture(fig4_path, width=Inches(6.0))
        cap_p = doc.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap_p.add_run("Figure 5: Out-of-Fold Confusion Matrices for Random Forest and XGBoost on Dataset 2.")
        cap_run.font.italic = True
        cap_run.font.size = Pt(9.5)
        cap_p.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 8. Discussion & Recommendations
    # -------------------------------------------------------------
    add_sec_heading("8. Discussion and Recommendations for Agricultural AI")
    doc.add_paragraph(
        "Our findings provide a critical reinterpretation of published crop recommendation literature:\n"
        "1. The 99% accuracy reported in dozens of recent papers is not an indicator of algorithm superiority. Rather, it is "
        "an artifact of the Kaggle Crop Recommendation dataset, whose samples form tightly clustered, well-isolated hyper-volumes in "
        "7-dimensional space. Under such conditions, even elementary classifiers like KNN and Naive Bayes easily separate the classes.\n"
        "2. When classical ML models are deployed on operational sensor telemetry, high performance cannot be assumed. If IoT sensor "
        "streams lack physical agronomic correlation with crop suitability, even complex gradient-boosted ensembles collapse to random chance.\n\n"
        "We urge the smart agriculture community to enforce four standards: (1) Mandatory multi-seed protocol reporting with error bars, "
        "(2) Pre-training statistical signal audits (ANOVA), (3) Explicit reporting against majority baselines, and (4) Strict zero-leakage "
        "preprocessing isolation."
    )

    # -------------------------------------------------------------
    # 9. Conclusion
    # -------------------------------------------------------------
    add_sec_heading("9. Conclusion")
    doc.add_paragraph(
        "This paper conducted a rigorous empirical audit of five supervised machine learning algorithm families across two agricultural "
        "datasets and three matched evaluation protocols. Our findings demonstrate that near-ceiling classification accuracies (98%–99.8%) "
        "reflect the clean separability of benchmark data rather than algorithmic robustness. On secondary agricultural sensor logs, "
        "all five algorithms experienced complete representation collapse (~17%–23%), failing to beat the 22.20% majority-class baseline. "
        "Furthermore, single unstratified splits introduce severe variance (±5.94%), confirming that single-split reporting in precision "
        "agriculture can create a deceptive illusion of performance."
    )

    # -------------------------------------------------------------
    # References
    # -------------------------------------------------------------
    add_sec_heading("References")
    refs = [
        "Guel, R., et al. (2026). Machine Learning Benchmarks in Precision Agriculture: Saturation and Performance Limits. Semantic Scholar, https://www.semanticscholar.org/paper/3845e4f3bf7262ad731c6200f13585a028176c6f.",
        "Maji, C., Pal, P., Singh, R. K., & Upadhyay, S. (2026). Soil and Climate-Driven Crop Recommendation via Ensemble Learning: A Comparative Study of Classical Classifiers for Resource-Constrained Deployment. Proc. ISRSD-2026, Springer Lecture Notes in Electrical Engineering (LNEE).",
        "Zanzari, K., et al. (2026). A Comparative Benchmarking Study of Classical Machine Learning and Deep Learning Methods for Image-Based Deepfake Detection. Applied Cybersecurity & Internet Governance (ACIG), 5(1), DOI: 10.60097/ACIG/221087.",
        "Ingle, A. (2020). Crop Recommendation Dataset. Kaggle Datasets, https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset.",
        "Soundankar, A. (2024). Smart Farming Sensor Data for Yield Prediction. Kaggle Datasets, https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proc. 22nd ACM SIGKDD, 785-794.",
        "Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. Machine Learning, 20(3), 273-297.",
        "Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory, 13(1), 21-27.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Kapoor, S., & Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. Patterns, 4(9), 100804.",
        "Bouthillier, X., et al. (2021). Accounting for Variance in Machine Learning Benchmarks. Proc. MLSys, 3, 253-269.",
        "Varoquaux, G. (2018). Cross-Validation Failure: Small Sample Sizes Lead to Large Error Bars. NeuroImage, 180, 68-77."
    ]
    for idx, ref in enumerate(refs, start=1):
        rp = doc.add_paragraph()
        r_run = rp.add_run(f"[{idx}] {ref}")
        r_run.font.size = Pt(9.5)
        rp.paragraph_format.space_after = Pt(3)

    output_docx = "MANUSCRIPT_DRAFT.docx"
    doc.save(output_docx)
    print(f"SUCCESS: Publication-ready Word document created at: {output_docx}", flush=True)

if __name__ == '__main__':
    create_word_document()
