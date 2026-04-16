# QUICKSTART — One-Day Execution Plan

You have a complete, runnable project. Follow these steps in order.

---

## Hour 1 — Get the real dataset and run the notebook

1. **Download the real dataset.** Go to https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset, sign in (free), download `WA_Fn-UseC_-HR-Employee-Attrition.csv` and **replace** the placeholder file in this folder. The current CSV is a synthetic test dataset — do **not** submit results from it.

2. **Set up your Anaconda environment** (following your course guidelines):
   ```
   conda create -n streamlit_env python=3.10
   conda activate streamlit_env
   pip install -r requirements.txt
   pip install jupyter
   ```

3. **Run the notebook.** Open `pipeline.ipynb` in Jupyter Notebook and run all cells (Kernel → Restart & Run All). Takes ~3 minutes. This produces:
   - `best_model.pkl` — trained pipeline
   - `metadata.pkl` — feature schema and tuned threshold
   - `metrics.json` — held-out test metrics
   - `figures/` — all 11 EDA and evaluation plots (refreshed)

4. **Take screenshots while you're in there** — you'll need 4–6 for Appendix A of the report. Suggested screenshots:
   - The `ColumnTransformer` + `Pipeline` definition cell
   - The dictionary of five candidate model pipelines
   - The `cross_val_score` loop and its output
   - The `GridSearchCV` cell + `best_params_` output
   - The threshold-tuning code
   - The SHAP `TreeExplainer` / `summary_plot` cell

## Hour 2 — Test the Streamlit app locally

```
conda activate streamlit_env
streamlit run app.py
```

Open http://localhost:8501, click through all four tabs:
- **Predict** — fill in sample inputs, click Predict
- **Explain** — see the SHAP per-prediction plot
- **Model insights** — see ROC/PR curves, confusion matrix, etc.
- **About** — problem description and limitations

**Take screenshots of each tab** — you'll need them for Section 5 of the report.

## Hour 3 — Push to GitHub & deploy to Streamlit Cloud

1. Create a **public** GitHub repo (e.g. `employee-attrition-com763`).
2. From this folder:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
3. Go to https://share.streamlit.io, sign in with GitHub.
4. **New app** → pick the repo → branch `main` → main file `app.py` → **Deploy**.
5. First deploy takes ~5 minutes. Once live, **copy the URL**.
6. Paste the URL on the cover page of the report and in the README.
7. Test from your phone or another browser to confirm it works.

## Hours 4–9 — Write the report

1. Open `Report_Skeleton.docx`. Structure is complete — every section has pre-written prose, all figures are embedded, and blue tip boxes explain what each section is graded on.
2. **Find every `[YOUR VALUE]` and `[REPLACE WITH ...]`** — fill in actual numbers from `metrics.json` and your notebook output.
3. **Find every yellow 📷 box** — replace with screenshots (~12 total: code snippets + Streamlit app screenshots).
4. **Personalise**: name, student ID, date, GitHub URL, Streamlit URL on the cover page.
5. **Edit §3.6 iteration log** — keep or modify the example iteration entries to match your actual experience. The rubric explicitly rewards honest evidence of debugging.
6. **Delete all blue tip boxes** before submitting (Ctrl+F "Rubric weight" to find them).

## Hour 10 — Final polish & submission

1. Read through end-to-end once. Check every figure renders and no placeholder text remains.
2. Export to PDF: File → Export → Create PDF/XPS Document.
3. Submit the PDF via the Turnitin link in Moodle.

---

## Flat file layout (what goes where)

Everything lives in the **same folder** — no subfolders except `figures/`:

| File | What it is | Committed to GitHub? |
|------|------------|---------------------|
| `app.py` | Streamlit application | Yes |
| `pipeline.ipynb` | Jupyter notebook — full ML pipeline | Yes |
| `best_model.pkl` | Trained sklearn Pipeline (pickle) | Yes |
| `metadata.pkl` | Feature schema + threshold (pickle) | Yes |
| `metrics.json` | Test-set metrics | Yes |
| `cv_summary.csv` | Cross-validation results | Yes |
| `requirements.txt` | Dependencies | Yes |
| `README.md` | Project README | Yes |
| `.gitignore` | Standard Python gitignore | Yes |
| `figures/*.png` | All EDA and evaluation plots (11 files) | Yes |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | Dataset | Yes |
| `Report_Skeleton.docx` | Report template (for your use, not the app) | Optional |
| `make_synthetic.py` | Generates test data (for dev only) | No |
| `make_diagram.py` | Generates architecture diagram | No |
| `build_report.js` | Builds the .docx report | No |

## If something breaks

- **"FileNotFoundError for best_model.pkl"** — the model file isn't in the same folder as `app.py`. Re-run `pipeline.ipynb` (all cells) to regenerate it.
- **Streamlit Cloud deploy fails** — check build logs. Most common: missing package in `requirements.txt`. Add it, push, app auto-redeploys.
- **Numbers look unrealistically perfect** — you ran on the synthetic dataset. Replace the CSV with the real Kaggle one and rerun.
- **SHAP is slow on the Explain tab** — this is normal for `KernelExplainer` (used with LogReg/SVM). If the winning model is XGBoost or RandomForest, SHAP uses the much faster `TreeExplainer` automatically.

Good luck.
