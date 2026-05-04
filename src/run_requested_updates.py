from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, learning_curve

from src.evaluate import compute_per_class_metrics, get_predictions
from src.features import load_vectorizer, transform_texts
from src.models import build_logistic_regression, load_model


def merge_roc_auc(results_dir: Path) -> pd.DataFrame:
    comparison_path = results_dir / "comparison_results.csv"
    roc_auc_path = results_dir / "roc_auc_scores.csv"

    comparison_df = pd.read_csv(comparison_path)
    roc_df = pd.read_csv(roc_auc_path)

    merged_df = comparison_df.merge(roc_df, on="model", how="left")
    merged_df = merged_df.sort_values(by="f1", ascending=False).reset_index(drop=True)
    merged_df.to_csv(comparison_path, index=False)
    return merged_df


def generate_per_class_heatmap(root: Path, results_dir: Path, figures_dir: Path) -> pd.DataFrame:
    test_df = pd.read_csv(root / "data/processed/test_data.csv")
    test_df["cleaned_text"] = test_df["cleaned_text"].fillna("").astype(str)

    vectorizer = load_vectorizer(root / "models/tfidf_vectorizer.pkl")
    x_test = transform_texts(test_df["cleaned_text"], vectorizer)
    y_test = test_df["class"]

    model_names = [
        "logistic_regression",
        "naive_bayes",
        "linear_svm",
        "calibrated_svm",
        "voting_classifier",
    ]

    rows = []
    for model_name in model_names:
        model = load_model(root / f"models/{model_name}.pkl")
        y_pred = get_predictions(model, x_test)
        class_df = compute_per_class_metrics(
            y_true=y_test,
            y_pred=y_pred,
            labels=["non-suicide", "suicide"],
        )
        class_df.insert(0, "model", model_name)
        rows.append(class_df)

    per_class_df = pd.concat(rows, ignore_index=True)
    per_class_df.to_csv(results_dir / "per_class_metrics.csv", index=False)

    heatmap_df = per_class_df.pivot(index="model", columns="class", values="f1")
    heatmap_df = heatmap_df.reindex(model_names)

    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="Blues", cbar=True)
    plt.title("Per-Class F1 Heatmap (Test Set)")
    plt.xlabel("Class")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "per_class_f1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    return per_class_df


def generate_classical_loss_curve(root: Path, figures_dir: Path, results_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(root / "data/processed/train_data.csv")
    train_df["cleaned_text"] = train_df["cleaned_text"].fillna("").astype(str)

    vectorizer = load_vectorizer(root / "models/tfidf_vectorizer.pkl")
    x_train = transform_texts(train_df["cleaned_text"], vectorizer)
    y_train = train_df["class"]

    model = build_logistic_regression()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, pos_label="suicide")

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=x_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=cv,
        scoring=scorer,
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_loss = 1.0 - train_mean
    val_loss = 1.0 - val_mean

    curve_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_f1": train_mean,
            "val_f1": val_mean,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
    )
    curve_df.to_csv(results_dir / "training_validation_curve.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_loss, marker="o", label="Training Loss (1 - F1)")
    plt.plot(train_sizes, val_loss, marker="o", label="Validation Loss (1 - F1)")
    plt.title("Classical Model Learning Curve (Logistic Regression)")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "training_validation_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    return curve_df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged_df = merge_roc_auc(results_dir)
    per_class_df = generate_per_class_heatmap(root, results_dir, figures_dir)
    curve_df = generate_classical_loss_curve(root, figures_dir, results_dir)

    print("Updated:", results_dir / "comparison_results.csv")
    print("Saved:", results_dir / "per_class_metrics.csv")
    print("Saved:", figures_dir / "per_class_f1_heatmap.png")
    print("Saved:", figures_dir / "training_validation_loss_curve.png")
    print("Saved:", results_dir / "training_validation_curve.csv")
    print("\nTop rows of comparison_results.csv:")
    print(merged_df.head().to_string(index=False))
    print("\nPer-class metrics rows:", len(per_class_df))
    print("Learning-curve points:", len(curve_df))


if __name__ == "__main__":
    main()
