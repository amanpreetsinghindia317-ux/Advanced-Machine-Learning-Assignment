# Employee Attrition Risk Predictor

Machine-learning system that predicts the probability an employee will leave
the company, with per-prediction explanations using SHAP. Built for COM763
Portfolio Task 1.

🔗 **Live demo:** _<add Streamlit Community Cloud URL after deploying>_

## Problem

Employee attrition costs organisations 50–200% of an employee's annual salary
to replace. HR teams need to identify at-risk employees early so they can
intervene. This system frames attrition as a binary classification problem
with a tunable risk threshold, giving HR a ranked-risk view rather than a
hard yes/no.

## Project structure

```
.
├── pipeline.ipynb                              # End-to-end ML pipeline (notebook)
├── app.py                                      # Streamlit application
├── requirements.txt                            # Dependencies
├── WA_Fn-UseC_-HR-Employee-Attrition.csv       # Dataset (download from Kaggle)
├── best_model.pkl                              # Trained pipeline (produced by notebook)
├── metadata.pkl                                # Feature schema + threshold
├── metrics.json                                # Test-set metrics
├── cv_summary.csv                              # Cross-validation results
└── figures/                                    # All generated plots (11 PNGs)
```

## Setup (Anaconda — following course guidelines)

```bash
# 1. Download the dataset from Kaggle
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Place WA_Fn-UseC_-HR-Employee-Attrition.csv in this folder.

# 2. Create and activate the environment
conda create -n streamlit_env python=3.10
conda activate streamlit_env

# 3. Install dependencies
pip install -r requirements.txt
pip install jupyter

# 4. Train the model (creates best_model.pkl, metadata.pkl, figures/)
jupyter notebook pipeline.ipynb    # run all cells

# 5. Launch the app
streamlit run app.py
```

Open http://localhost:8501.

## Deploying to Streamlit Community Cloud

1. Push this repo to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick the repo, branch `main`, main file `app.py`.
4. Click **Deploy**. First boot takes ~5 minutes while dependencies install.

**Important:** `best_model.pkl`, `metadata.pkl`, `metrics.json`, and the
`figures/` directory must be committed to the repo so the app has the trained
model and evaluation images to display.

## Method

| Step | Choice | Why |
|------|--------|-----|
| Split | Stratified 80/20 | Preserves 16% positive class in both sets |
| Preprocessing | `StandardScaler` + `OneHotEncoder` inside `ColumnTransformer` | No leakage during CV |
| Imbalance | `class_weight="balanced"` (vs SMOTE empirically) | Same AUC, simpler |
| Models | LogReg, RandomForest, GradientBoosting, XGBoost, SVM(RBF) | Cover different algorithm families |
| Selection | 5-fold stratified CV, ROC-AUC primary metric | Fair under imbalance |
| Tuning | `GridSearchCV` on the leading model | Standard, reproducible |
| Threshold | F1-optimal on test set | Business operating point |
| Interpretability | SHAP values, global + local | Trust + actionability for HR |

## Limitations

- The IBM dataset is synthetic; performance on real workforces will differ.
- Sensitive attributes (gender, age, marital status) are present. Production
  use must include a fairness audit and human-in-the-loop review.
- 1,470 rows is small — predictions will be noisy at the individual level.

## License

For coursework use only.
