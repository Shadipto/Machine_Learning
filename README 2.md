# Suicide Risk Text Classification

This project builds a text classification pipeline for the `Suicide_Detection.csv` dataset.

## Current Status

- Phase 0 setup is complete in practice.
- The `mental_health_nlp` conda environment exists and is active.
- Core dependencies from `requirements.txt` are installed in that environment.
- The raw dataset loads successfully and has been inspected.

## Verified Dataset Summary

- File: `data/raw/Suicide_Detection.csv`
- Shape: `232074 x 3`
- Columns: `Unnamed: 0`, `text`, `class`
- Missing values: none
- Class balance: `suicide` 50%, `non-suicide` 50%

## Environment

Use the existing conda environment:

```bash
conda activate mental_health_nlp
```

Useful checks:

```bash
which python
python --version
python -m pip show pandas scikit-learn numpy nltk imbalanced-learn matplotlib seaborn joblib
```

## Repository Layout

- `data/raw/` - original dataset
- `data/processed/` - cleaned and split data artifacts
- `models/` - saved vectorizers and model files
- `notebooks/` - exploration and development notebooks
- `src/` - reusable preprocessing, feature, model, and evaluation code
- `results/` - plots and metrics outputs

## Current Artifacts

- Cleaned dataset: `data/processed/cleaned_data.csv`
- Train split: `data/processed/train_data.csv`
- Validation split: `data/processed/validation_data.csv`
- Test split: `data/processed/test_data.csv`
- TF-IDF vectorizer: `models/tfidf_vectorizer.pkl`
- Trained models: `models/logistic_regression.pkl`, `models/naive_bayes.pkl`, `models/linear_svm.pkl`, `models/calibrated_svm.pkl`, `models/voting_classifier.pkl`
- Evaluation table: `results/comparison_results.csv`
- Evaluation figures: `results/figures/`
- Tier 1 ROC-AUC scores: `results/roc_auc_scores.csv`
- Tier 1 feature-importance table: `results/feature_importance.csv`
- Tier 1 ROC figure: `results/figures/roc_auc_curves.png`
- Tier 1 feature-importance figure: `results/figures/feature_importance.png`

## Planned Workflow

1. Explore the raw dataset.
2. Clean and normalize the text.
3. Save a cleaned dataset in `data/processed/`.
4. Split the data into train, validation, and test sets.
5. Build TF-IDF features from the training split.
6. Train baseline models.
7. Evaluate models and save comparison outputs.
8. Add inference support and final documentation.

## How To Reproduce

1. Activate the environment with `conda activate mental_health_nlp`.
2. Open and run the notebooks in order:
   1. `notebooks/01_data_exploration.ipynb`
   2. `notebooks/02_preprocessing.ipynb`
   3. `notebooks/03_feature_extraction.ipynb`
   4. `notebooks/04_models.ipynb`
   5. `notebooks/05_evaluation.ipynb`
   6. `notebooks/06_inference.ipynb`
3. Review the generated files in `data/processed/`, `models/`, and `results/`.
4. Use the `predict_text()` helper in the inference notebook for quick manual checks.

## Tier 1 Implementation Update (2026-04-30)

Completed Tier 1 tasks from `coding_checklist.md`:

- Implemented reusable evaluation module in `src/evaluate.py`:
  - model prediction and metric helpers
  - confusion matrix generation
  - ROC-AUC generation for single and multiple models
  - TF-IDF feature-importance plotting
- Added Tier 1 artifact runner: `src/run_tier1_artifacts.py`
- Generated missing Tier 1 outputs:
  - `results/figures/roc_auc_curves.png`
  - `results/figures/feature_importance.png`
  - `results/roc_auc_scores.csv`
  - `results/feature_importance.csv`
- Pinned project dependencies in `requirements.txt` using versions from the active `mental_health_nlp` environment.

Run command used for Tier 1 artifact generation:

```bash
conda run -n mental_health_nlp python -m src.run_tier1_artifacts
```

## Running Notes

This section will be updated as the project moves forward so the work stays documented in one place.

### 2026-04-24

- Confirmed the conda environment exists and is active.
- Verified the main ML dependencies are installed.
- Loaded the raw dataset successfully in the exploration notebook.
- Confirmed the dataset has no missing values.
- Confirmed the dataset is perfectly balanced across the two classes.
- Noted that `Unnamed: 0` appears to be an index-style column and can likely be dropped during preprocessing.

### 2026-04-24 - Phase 1 started

- Began `notebooks/02_preprocessing.ipynb`.
- Inspected the raw `text` column for missing values, blank strings, and length distribution.
- Confirmed the text column has no missing or blank values.
- Defined the first reusable cleaning function: lowercase, URL removal, HTML cleanup, punctuation removal, digit removal, and whitespace normalization.
- Next: test the cleaning function on more samples and then move it into `src/preprocess.py`.

### 2026-04-24 - Cleaning applied

- Applied the simple cleaning function to the full dataset.
- Dropped the `Unnamed: 0` column.
- Saved a cleaned dataset to `data/processed/cleaned_data.csv`.
- Kept both `text` and `cleaned_text` columns for now so we can compare them in the next step.
- Next: inspect the cleaned output briefly and then start the train/validation/test split.

### 2026-04-24 - Cleaned data validated

- Added a quick validation step in `notebooks/02_preprocessing.ipynb`.
- Confirmed the cleaned dataset still has the expected shape and columns.
- Reviewed a few original vs cleaned text samples before moving on.
- Next: start the train/validation/test split with stratification.

### 2026-04-24 - Split started

- Began the stratified train/validation/test split in `notebooks/02_preprocessing.ipynb`.
- Planned to save `train_data.csv`, `validation_data.csv`, and `test_data.csv` in `data/processed/`.
- Keeping the class balance intact during splitting.
- Next: verify the split sizes and class counts, then move toward feature extraction.

### 2026-04-24 - Feature extraction started

- Began `notebooks/03_feature_extraction.ipynb`.
- Loaded the saved train, validation, and test splits.
- Fit a TF-IDF vectorizer on the training split only.
- Transformed the validation and test splits with the same fitted vectorizer.
- Saved the vectorizer to `models/tfidf_vectorizer.pkl`.
- Next: move on to model training.

### 2026-04-24 - Model training started

- Began `notebooks/04_models.ipynb`.
- Loaded the saved TF-IDF vectorizer and processed splits.
- Set up baseline classifiers for validation comparison.
- Next: train the models, save the fitted artifacts, and compare validation metrics.

### 2026-04-24 - Model training completed

- Trained logistic regression, naive Bayes, linear SVM, calibrated SVM, and a hard voting classifier.
- Saved fitted model artifacts in `models/`.
- Compared the models on the validation split.
- Best validation result so far: `voting_classifier` with F1 around `0.940`.
- Next: move on to evaluation on the held-out test split and generate comparison outputs.

### 2026-04-24 - Evaluation started

- Began `notebooks/05_evaluation.ipynb`.
- Loaded the saved models and the test split.
- Planned to compute test metrics, confusion matrices, and a comparison table.
- Next: run the evaluation and save the figures and metrics table.

### 2026-04-24 - Evaluation completed

- Evaluated the saved models on the held-out test split.
- Saved confusion matrices in `results/figures/`.
- Saved the comparison table to `results/comparison_results.csv`.
- Best test result so far: `voting_classifier` with F1 around `0.941`.
- Next: create the inference pipeline and then finalize documentation.

### 2026-04-24 - Inference notebook created

- Created `notebooks/06_inference.ipynb`.
- The notebook loads the saved vectorizer and best model.
- Added a `predict_text()` helper for raw text input.
- Added a few sample prompts for quick manual testing.
- Next: run the notebook cells and make sure the predictions work end to end.

### 2026-04-24 - Inference notebook validated

- Ran `notebooks/06_inference.ipynb` end to end.
- Confirmed the saved vectorizer and `voting_classifier` model load correctly.
- Confirmed `predict_text()` returns labels on sample raw text inputs.
- Next: finalize documentation and clean up any leftover notebook or code notes.

### 2026-04-24 - Phase 7 started

- Added a reproducible project summary and artifact list to `README.md`.
- Documented the notebook execution order for rerunning the project from scratch.
- Next: finish any remaining cleanup notes and keep this README updated if more artifacts are added.

### 2026-04-24 - Phase 7 completed

- Identified the stray `.DS_Store` file at the project root; it is harmless macOS metadata.
- Confirmed there are no notebook checkpoint or `__pycache__` folders in the workspace.
- Kept the README as the main project record for artifacts, reproduction steps, and phase history.
- The project is now documented through the inference stage and cleanup pass.

### 2026-04-30 - Tier 1 completed

- Implemented `src/evaluate.py` with reusable evaluation and plotting utilities.
- Added `src/run_tier1_artifacts.py` to generate required Tier 1 outputs in one command.
- Generated ROC-AUC and feature-importance artifacts in `results/`.
- Replaced the previous environment dump in `requirements.txt` with a clean pinned dependency list.

## Next Step

Proceed with Tier 2 tasks in `coding_checklist.md` (module extraction for `src/preprocess.py`, `src/features.py`, `src/models.py`, and error-analysis tooling), then start neural network implementation.
