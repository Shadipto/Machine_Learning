from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	auc,
	classification_report,
	f1_score,
	precision_score,
	recall_score,
	roc_curve,
)


def get_predictions(model: Any, X: Any) -> np.ndarray:
	"""Return model predictions for feature matrix X."""
	return np.asarray(model.predict(X))


def calculate_metrics(
	y_true: pd.Series | np.ndarray,
	y_pred: pd.Series | np.ndarray,
	positive_label: str = "suicide",
) -> dict[str, float]:
	"""Compute standard binary classification metrics."""
	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred, pos_label=positive_label),
		"recall": recall_score(y_true, y_pred, pos_label=positive_label),
		"f1": f1_score(y_true, y_pred, pos_label=positive_label),
	}


def get_classification_report(
	y_true: pd.Series | np.ndarray,
	y_pred: pd.Series | np.ndarray,
) -> str:
	"""Return sklearn classification report as a string."""
	return classification_report(y_true, y_pred)


def compare_models(
	models: dict[str, Any],
	X_test: Any,
	y_test: pd.Series | np.ndarray,
	positive_label: str = "suicide",
) -> pd.DataFrame:
	"""Evaluate many models on the same test set and return a sorted table."""
	rows: list[dict[str, float | str]] = []
	for model_name, model in models.items():
		y_pred = get_predictions(model, X_test)
		metrics = calculate_metrics(y_test, y_pred, positive_label=positive_label)
		rows.append({"model": model_name, **metrics})
	return pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)


def save_comparison_table(results_df: pd.DataFrame, output_path: str | Path) -> None:
	"""Save the model comparison table as CSV."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	results_df.to_csv(output_path, index=False)


def generate_confusion_matrix(
	y_true: pd.Series | np.ndarray,
	y_pred: pd.Series | np.ndarray,
	model_name: str,
	output_path: str | Path,
	label_order: list[str] | None = None,
) -> None:
	"""Save a confusion-matrix plot for one model."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	fig, ax = plt.subplots(figsize=(5, 4))
	ConfusionMatrixDisplay.from_predictions(
		y_true,
		y_pred,
		labels=label_order,
		display_labels=label_order,
		cmap="Blues",
		ax=ax,
		colorbar=False,
	)
	ax.set_title(f"{model_name} confusion matrix")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def _to_binary_labels(
	y_true: pd.Series | np.ndarray,
	positive_label: str,
) -> np.ndarray:
	y_true_array = np.asarray(y_true)
	return (y_true_array == positive_label).astype(int)


def _model_scores(
	model: Any,
	X: Any,
	positive_label: str = "suicide",
) -> np.ndarray:
	"""Get continuous scores for ROC where possible, with safe fallbacks."""
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(X)
		if proba.ndim == 2 and proba.shape[1] >= 2:
			classes = list(getattr(model, "classes_", []))
			if positive_label in classes:
				return proba[:, classes.index(positive_label)]
			return proba[:, -1]

	if hasattr(model, "decision_function"):
		decision = model.decision_function(X)
		decision = np.asarray(decision)
		if decision.ndim > 1:
			return decision[:, -1]
		return decision

	# Fallback for classifiers that expose only hard labels.
	return (np.asarray(model.predict(X)) == positive_label).astype(float)


def generate_roc_auc_curve(
	y_true: pd.Series | np.ndarray,
	y_scores: pd.Series | np.ndarray,
	model_name: str,
	output_path: str | Path,
	positive_label: str = "suicide",
) -> float:
	"""Generate and save a single ROC curve; returns AUC."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	y_binary = _to_binary_labels(y_true, positive_label=positive_label)
	fpr, tpr, _ = roc_curve(y_binary, np.asarray(y_scores))
	roc_auc = auc(fpr, tpr)

	fig, ax = plt.subplots(figsize=(6, 5))
	ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
	ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title(f"ROC Curve - {model_name}")
	ax.legend(loc="lower right")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	return roc_auc


def generate_roc_auc_curves_for_models(
	models: dict[str, Any],
	X_test: Any,
	y_test: pd.Series | np.ndarray,
	output_path: str | Path,
	positive_label: str = "suicide",
) -> pd.DataFrame:
	"""Generate a combined ROC figure for multiple models and return AUC table."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	y_binary = _to_binary_labels(y_test, positive_label=positive_label)
	fig, ax = plt.subplots(figsize=(8, 6))
	auc_rows: list[dict[str, float | str]] = []

	for model_name, model in models.items():
		scores = _model_scores(model, X_test, positive_label=positive_label)
		fpr, tpr, _ = roc_curve(y_binary, scores)
		roc_auc = auc(fpr, tpr)
		auc_rows.append({"model": model_name, "roc_auc": float(roc_auc)})
		ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={roc_auc:.4f})")

	ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title("ROC-AUC Comparison")
	ax.legend(loc="lower right", fontsize=8)
	fig.tight_layout()
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)

	return pd.DataFrame(auc_rows).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)


def generate_feature_importance(
	X_text: sparse.spmatrix,
	y_true: pd.Series | np.ndarray,
	vectorizer: Any,
	output_path: str | Path,
	top_n: int = 20,
	positive_label: str = "suicide",
) -> pd.DataFrame:
	"""Plot top TF-IDF features by class-wise mean score and return the table."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	y_array = np.asarray(y_true)
	feature_names = np.asarray(vectorizer.get_feature_names_out())

	pos_mask = y_array == positive_label
	neg_mask = ~pos_mask

	if pos_mask.sum() == 0 or neg_mask.sum() == 0:
		raise ValueError("Both classes must be present to compute feature importance.")

	pos_mean = np.asarray(X_text[pos_mask].mean(axis=0)).ravel()
	neg_mean = np.asarray(X_text[neg_mask].mean(axis=0)).ravel()
	diff = pos_mean - neg_mean

	top_pos_idx = np.argsort(diff)[-top_n:][::-1]
	top_neg_idx = np.argsort(diff)[:top_n]

	pos_df = pd.DataFrame(
		{
			"feature": feature_names[top_pos_idx],
			"score": diff[top_pos_idx],
			"class": positive_label,
		}
	)
	neg_label = "non-suicide"
	neg_df = pd.DataFrame(
		{
			"feature": feature_names[top_neg_idx],
			"score": np.abs(diff[top_neg_idx]),
			"class": neg_label,
		}
	)
	importance_df = pd.concat([pos_df, neg_df], ignore_index=True)

	fig, axes = plt.subplots(1, 2, figsize=(14, 8))
	sns.barplot(data=pos_df, x="score", y="feature", ax=axes[0], color="crimson")
	axes[0].set_title(f"Top {top_n} Features: {positive_label}")
	axes[0].set_xlabel("Mean TF-IDF Difference")
	axes[0].set_ylabel("Feature")

	sns.barplot(data=neg_df, x="score", y="feature", ax=axes[1], color="steelblue")
	axes[1].set_title(f"Top {top_n} Features: {neg_label}")
	axes[1].set_xlabel("Mean TF-IDF Difference")
	axes[1].set_ylabel("Feature")

	fig.tight_layout()
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)

	return importance_df


def generate_error_analysis(
	y_true: pd.Series | np.ndarray,
	y_pred: pd.Series | np.ndarray,
	texts: pd.Series | np.ndarray,
	model_name: str,
	output_path: str | Path,
	positive_label: str = "suicide",
	max_rows: int | None = None,
) -> pd.DataFrame:
	"""Create and save a misclassification table for qualitative error inspection."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	y_true_arr = np.asarray(y_true)
	y_pred_arr = np.asarray(y_pred)
	texts_arr = np.asarray(texts)

	error_mask = y_true_arr != y_pred_arr
	error_df = pd.DataFrame(
		{
			"text": texts_arr[error_mask],
			"true_label": y_true_arr[error_mask],
			"predicted_label": y_pred_arr[error_mask],
			"model": model_name,
		}
	)

	error_df["error_type"] = np.where(
		(error_df["true_label"] == positive_label) & (error_df["predicted_label"] != positive_label),
		"false_negative",
		"false_positive",
	)

	if max_rows is not None and max_rows > 0:
		error_df = error_df.head(max_rows)

	error_df.to_csv(output_path, index=False)
	return error_df


def compute_per_class_metrics(
	y_true: pd.Series | np.ndarray,
	y_pred: pd.Series | np.ndarray,
	labels: list[str] | None = None,
) -> pd.DataFrame:
	"""Return per-class precision/recall/F1/support as a tidy DataFrame."""
	labels_list = labels if labels is not None else sorted(pd.Series(y_true).unique().tolist())
	report_dict = classification_report(
		y_true,
		y_pred,
		labels=labels_list,
		output_dict=True,
		zero_division=0,
	)

	rows: list[dict[str, float | str]] = []
	for label in labels_list:
		row = report_dict.get(label, {})
		rows.append(
			{
				"class": label,
				"precision": float(row.get("precision", 0.0)),
				"recall": float(row.get("recall", 0.0)),
				"f1": float(row.get("f1-score", 0.0)),
				"support": float(row.get("support", 0.0)),
			}
		)

	return pd.DataFrame(rows)
