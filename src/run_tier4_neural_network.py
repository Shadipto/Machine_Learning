from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate import (
    calculate_metrics,
    compute_per_class_metrics,
    generate_confusion_matrix,
    generate_error_analysis,
    generate_roc_auc_curve,
)
from src.features import load_vectorizer, transform_texts
from src.nn_model import SparseBinaryMLP, TrainingHistory, save_sparse_mlp


def _history_to_dataframe(history: list[TrainingHistory]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch": [h.epoch for h in history],
            "train_loss": [h.train_loss for h in history],
            "val_loss": [h.val_loss for h in history],
            "train_f1": [h.train_f1 for h in history],
            "val_f1": [h.val_f1 for h in history],
        }
    )


def _plot_neural_loss_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation Loss")
    ax.set_title("Neural MLP Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _update_comparison_table(comparison_path: Path, new_row: dict[str, float | str]) -> pd.DataFrame:
    comparison_df = pd.read_csv(comparison_path)
    comparison_df = comparison_df[comparison_df["model"] != new_row["model"]].copy()

    needed_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    for column in needed_cols:
        if column not in comparison_df.columns:
            comparison_df[column] = pd.NA

    updated = pd.concat([comparison_df, pd.DataFrame([new_row])], ignore_index=True)
    updated = updated.sort_values(by="f1", ascending=False).reset_index(drop=True)
    updated.to_csv(comparison_path, index=False)
    return updated


def _update_per_class_metrics(per_class_path: Path, neural_class_df: pd.DataFrame) -> pd.DataFrame:
    if per_class_path.exists():
        base_df = pd.read_csv(per_class_path)
        base_df = base_df[base_df["model"] != "neural_mlp"].copy()
    else:
        base_df = pd.DataFrame(columns=["model", "class", "precision", "recall", "f1", "support"])

    updated = pd.concat([base_df, neural_class_df], ignore_index=True)
    updated.to_csv(per_class_path, index=False)
    return updated


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    models_dir = root / "models"
    results_dir = root / "results"
    figures_dir = results_dir / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train_data.csv")
    val_df = pd.read_csv(data_dir / "validation_data.csv")
    test_df = pd.read_csv(data_dir / "test_data.csv")

    for split_df in (train_df, val_df, test_df):
        split_df["cleaned_text"] = split_df["cleaned_text"].fillna("").astype(str)

    vectorizer = load_vectorizer(models_dir / "tfidf_vectorizer.pkl")
    X_train = transform_texts(train_df["cleaned_text"], vectorizer)
    X_val = transform_texts(val_df["cleaned_text"], vectorizer)
    X_test = transform_texts(test_df["cleaned_text"], vectorizer)

    y_train = train_df["class"].values
    y_val = val_df["class"].values
    y_test = test_df["class"].values

    nn_model = SparseBinaryMLP(
        input_dim=X_train.shape[1],
        hidden_dim_1=256,
        hidden_dim_2=64,
        dropout_1=0.3,
        dropout_2=0.2,
        learning_rate=1e-3,
        l2_lambda=1e-5,
        batch_size=1024,
        max_epochs=20,
        patience=4,
        random_state=42,
        positive_label="suicide",
        negative_label="non-suicide",
    )

    train_start = time.perf_counter()
    history = nn_model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )
    training_time_sec = time.perf_counter() - train_start

    model_path = models_dir / "neural_mlp.pkl"
    save_sparse_mlp(nn_model, model_path)

    history_df = _history_to_dataframe(history)
    history_csv_path = results_dir / "neural_training_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    _plot_neural_loss_curve(history_df, figures_dir / "neural_loss_curve.png")

    infer_start = time.perf_counter()
    y_pred = nn_model.predict(X_test)
    y_scores = nn_model.predict_positive_proba(X_test)
    infer_total_sec = time.perf_counter() - infer_start

    inference_latency_ms_per_sample = (infer_total_sec / max(len(y_test), 1)) * 1000.0

    base_metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, positive_label="suicide")
    roc_auc = roc_auc_score((y_test == "suicide").astype(int), y_scores)

    generate_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        model_name="neural_mlp",
        output_path=figures_dir / "neural_mlp_confusion_matrix.png",
        label_order=["non-suicide", "suicide"],
    )

    generate_roc_auc_curve(
        y_true=y_test,
        y_scores=y_scores,
        model_name="neural_mlp",
        output_path=figures_dir / "neural_roc_curve.png",
        positive_label="suicide",
    )

    neural_row = {
        "model": "neural_mlp",
        "accuracy": float(base_metrics["accuracy"]),
        "precision": float(base_metrics["precision"]),
        "recall": float(base_metrics["recall"]),
        "f1": float(base_metrics["f1"]),
        "roc_auc": float(roc_auc),
    }

    comparison_df = _update_comparison_table(
        comparison_path=results_dir / "comparison_results.csv",
        new_row=neural_row,
    )

    neural_per_class_df = compute_per_class_metrics(
        y_true=y_test,
        y_pred=y_pred,
        labels=["non-suicide", "suicide"],
    )
    neural_per_class_df.insert(0, "model", "neural_mlp")
    updated_per_class_df = _update_per_class_metrics(
        per_class_path=results_dir / "per_class_metrics.csv",
        neural_class_df=neural_per_class_df,
    )

    generate_error_analysis(
        y_true=y_test,
        y_pred=y_pred,
        texts=test_df["text"].fillna("").astype(str),
        model_name="neural_mlp",
        output_path=results_dir / "error_analysis_neural_mlp.csv",
        positive_label="suicide",
    )

    efficiency_df = pd.DataFrame(
        [
            {
                "model": "neural_mlp",
                "training_time_sec": float(training_time_sec),
                "inference_latency_ms_per_sample": float(inference_latency_ms_per_sample),
                "test_samples": int(len(y_test)),
            }
        ]
    )
    efficiency_df.to_csv(results_dir / "neural_efficiency.csv", index=False)

    print("Saved:", model_path)
    print("Saved:", history_csv_path)
    print("Saved:", figures_dir / "neural_loss_curve.png")
    print("Saved:", figures_dir / "neural_roc_curve.png")
    print("Saved:", figures_dir / "neural_mlp_confusion_matrix.png")
    print("Saved:", results_dir / "error_analysis_neural_mlp.csv")
    print("Saved:", results_dir / "neural_efficiency.csv")
    print("Updated:", results_dir / "comparison_results.csv")
    print("Updated:", results_dir / "per_class_metrics.csv")
    print("\nNeural model metrics:")
    print(pd.DataFrame([neural_row]).to_string(index=False))
    print("\nTraining time (sec):", round(training_time_sec, 2))
    print("Inference latency (ms/sample):", round(inference_latency_ms_per_sample, 6))
    print("Per-class rows now:", len(updated_per_class_df))
    print("Top rows in updated comparison table:")
    print(comparison_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
