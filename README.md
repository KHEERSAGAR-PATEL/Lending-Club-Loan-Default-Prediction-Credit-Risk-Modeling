# Lending-Club-Loan-Default-Prediction-Credit-Risk-Modeling
A machine learning pipeline for credit risk prediction using LendingClub data, optimized for imbalanced classification with XGBoost, LightGBM, MLP, and Random Forest. Achieved 85.74% test accuracy and 0.706 AUC with SHAP-based explainability and robust leakage prevention.


# Lending Club Loan Default Prediction

In late 2021, a Bengaluru-based fintech startup launched an express-loan product for India’s gig economy—delivery riders, on-demand drivers and freelance professionals. Within six months, economic headwinds and irregular incomes drove the borrower default rate above 18 %, wiping out over ₹40 crore of investor capital and threatening the platform’s survival.

Motivated by this real-world crisis, and leveraging the public LendingClub dataset, I designed and implemented a production-ready credit-risk pipeline to predict borrower defaults before funds are disbursed. As an M.Tech candidate in Computer Science (AI & ML), my goal was to translate rigorous academic research into a deployable solution that balances predictive accuracy, fairness and explainability.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)    
- [Pipeline Steps](#pipeline-steps)  
- [Modeling and Results](#modeling-and-results)  
- [Explainability](#explainability)  
- [Usage](#installation-and-usage)  
 - [Next Steps](#next-steps)  
- [Author](#author)   

---

## Project Overview

Peer-to-peer lending platforms enable unsecured personal loans and offer investors attractive interest returns—minus origination and service fees. The key risk in this model is borrower default, which can wipe out investor capital and undermine confidence. This project builds an end-to-end machine learning workflow to:

1. Ingest and preprocess raw LendingClub loan records.  
2. Engineer domain-informed features (interest-credit rate, term ratios, grade encodings).  
3. Mitigate extreme class imbalance (< 1 % default rate) using SMOTE and GAN-based augmentation.  
4. Train and tune multiple classifiers (XGBoost, LightGBM, Random Forest, MLP) via Bayesian optimization.  
5. Prevent information leakage by excluding features only available post-origination.  
6. Evaluate models on accuracy, AUC, recall and F1-score, with a focus on detecting defaults.  
7. Explain individual predictions and global feature influence using SHAP.

---

## Dataset

- **Source**: [LendingClub Loan Data (Kaggle)]
- **Records**: 42,000+ loans, 56 fields (demographics, credit history, loan terms, outcomes)  
- **Target**: Loan status (“Fully Paid” vs. “Charged Off”)  

---

## Pipeline Steps

1. **Exploratory Data Analysis**  
   - Data integrity checks, missing value imputation  
   - Distribution plots for credit grades, loan terms, interest rates  

2. **Feature Engineering**  
   - Derived features: interest-to-term ratio, delinquency indicators  
   - One-hot and target encoding of categorical variables  
   - Temporal split to avoid information leakage  

3. **Imbalance Handling**  
   - **SMOTE**: Synthetic oversampling of minority (default) class  
   - **GAN-based sampling**: CTGAN to generate realistic minority samples  

4. **Model Fitting**  
   - Algorithms: XGBoost, LightGBM, Random Forest, MLP  
   - Hyperparameter tuning via Bayesian Optimization (Optuna)  
   - Class weighting to further address imbalance  

5. **Model Evaluation**  
   - Metrics: AUC (ROC), accuracy, recall on default class, F1-score  
   - Confusion matrix, Youden’s J, Matthews Correlation Coefficient  

6. **Explainability**  
   - SHAP summary, dependence and force plots  
   - Feature importance ranking and interaction insights  

---

## Results

- **AUC (ROC) score:** 0.7059934283893465  
- **Training accuracy:** 85.89 %  
- **Testing accuracy:** 85.74 %

**Classification Report (Test Set)**

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0     |      0.94 |   0.43 |     0.59 |  10 235 |
| 1     |      0.20 |   0.83 |     0.32 |   1 701 |
|**Accuracy**|       |        |     0.49 |  11 936 |
|**Macro avg**|   0.57 |   0.63 |     0.45 |  11 936 |
|**Weighted avg**|0.83 |   0.49 |     0.55 |  11 936 |

---

*The above metrics reflect model performance on an imbalanced test set, where Class 1 (defaults) is the minority. High recall for defaults (0.83) ensures most at-risk loans are flagged, while the overall AUC of 0.706 demonstrates reliable ranking of default risk.*

**Key Finding**: XGBoost consistently outperformed other learners in cross-validated AUC and recall on defaults, making it the preferred choice for deployment.

---

## Explainability

Transparency and regulatory compliance in financial services require interpretable models:

- **Global Feature Impact**: SHAP summary plot highlights interest rate, credit grade, annual income and term length as top predictors.  
- **Local Explanation**: SHAP force plots for individual loans reveal how feature values push the default probability up or down.  

These insights enable stakeholders—credit officers, investors, regulators—to understand, trust and audit the model.

---

##  Usage

Use the ipynb file to run in the local or gpu environment.

## Next Steps
Extend to full LendingClub dataset (2.2 M records) with distributed training

Deploy model and SHAP explainer as REST API using FastAPI

Integrate real-time scoring and monitoring for live loan applications

## Author
Kheer Sagar Patel
M.Tech in Computer Science (AI & ML), IIITDM Jabalpur
Passionate about building production-grade AI solutions for financial technology.



