from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.evaluate import generate_feature_importance, generate_roc_auc_curves_for_models


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    results_dir = root / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(root / "data/processed/test_data.csv")
    test_df["cleaned_text"] = test_df["cleaned_text"].fillna("").astype(str)
    y_test = test_df["class"]

    with open(models_dir / "tfidf_vectorizer.pkl", "rb") as fh:
        vectorizer = pickle.load(fh)

    X_test = vectorizer.transform(test_df["cleaned_text"])

    model_names = [
        "logistic_regression",
        "naive_bayes",
        "linear_svm",
        "calibrated_svm",
        "voting_classifier",
    ]
    models = {}
    for name in model_names:
        with open(models_dir / f"{name}.pkl", "rb") as fh:
            models[name] = pickle.load(fh)

    roc_df = generate_roc_auc_curves_for_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        output_path=figures_dir / "roc_auc_curves.png",
        positive_label="suicide",
    )
    roc_df.to_csv(results_dir / "roc_auc_scores.csv", index=False)

    importance_df = generate_feature_importance(
        X_text=X_test,
        y_true=y_test,
        vectorizer=vectorizer,
        output_path=figures_dir / "feature_importance.png",
        top_n=20,
        positive_label="suicide",
    )
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

    print("Saved:", figures_dir / "roc_auc_curves.png")
    print("Saved:", figures_dir / "feature_importance.png")
    print("Saved:", results_dir / "roc_auc_scores.csv")
    print("Saved:", results_dir / "feature_importance.csv")


if __name__ == "__main__":
    main()
