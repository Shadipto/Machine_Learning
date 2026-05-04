from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluate import (
    compute_per_class_metrics,
    generate_error_analysis,
    get_predictions,
)
from src.features import load_vectorizer, transform_texts
from src.models import load_model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    results_dir = root / "results"

    test_df = pd.read_csv(processed_dir / "test_data.csv")
    test_df["cleaned_text"] = test_df["cleaned_text"].fillna("").astype(str)

    vectorizer = load_vectorizer(root / "models" / "tfidf_vectorizer.pkl")
    model = load_model(root / "models" / "voting_classifier.pkl")

    x_test = transform_texts(test_df["cleaned_text"], vectorizer)
    y_true = test_df["class"]
    y_pred = get_predictions(model, x_test)

    error_df = generate_error_analysis(
        y_true=y_true,
        y_pred=y_pred,
        texts=test_df["text"],
        model_name="voting_classifier",
        output_path=results_dir / "error_analysis.csv",
        positive_label="suicide",
    )

    per_class_df = compute_per_class_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=["non-suicide", "suicide"],
    )
    per_class_df.to_csv(results_dir / "per_class_metrics.csv", index=False)

    print("Saved:", results_dir / "error_analysis.csv")
    print("Saved:", results_dir / "per_class_metrics.csv")
    print("Misclassified rows:", len(error_df))


if __name__ == "__main__":
    main()
