from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


DEFAULT_RANDOM_SEED = 42


def build_logistic_regression(
	max_iter: int = 1000,
	class_weight: str | dict[str, float] | None = "balanced",
	random_state: int = DEFAULT_RANDOM_SEED,
) -> LogisticRegression:
	"""Construct Logistic Regression baseline model."""
	return LogisticRegression(
		max_iter=max_iter,
		class_weight=class_weight,
		random_state=random_state,
	)


def build_naive_bayes() -> MultinomialNB:
	"""Construct Multinomial Naive Bayes baseline model."""
	return MultinomialNB()


def build_linear_svm(
	class_weight: str | dict[str, float] | None = "balanced",
	random_state: int = DEFAULT_RANDOM_SEED,
) -> LinearSVC:
	"""Construct LinearSVC baseline model."""
	return LinearSVC(class_weight=class_weight, random_state=random_state)


def build_calibrated_svm(
	class_weight: str | dict[str, float] | None = "balanced",
	random_state: int = DEFAULT_RANDOM_SEED,
	method: str = "sigmoid",
	cv: int = 3,
) -> CalibratedClassifierCV:
	"""Construct calibrated SVM model for probability-like outputs."""
	base_model = build_linear_svm(class_weight=class_weight, random_state=random_state)
	return CalibratedClassifierCV(base_model, method=method, cv=cv)


def build_voting_classifier(
	random_state: int = DEFAULT_RANDOM_SEED,
	voting: str = "hard",
) -> VotingClassifier:
	"""Construct hard-voting ensemble of LR + NB + LinearSVC."""
	return VotingClassifier(
		estimators=[
			("lr", build_logistic_regression(random_state=random_state)),
			("nb", build_naive_bayes()),
			("svm", build_linear_svm(random_state=random_state)),
		],
		voting=voting,
	)


def build_model_registry(random_state: int = DEFAULT_RANDOM_SEED) -> dict[str, Any]:
	"""Return all baseline model instances keyed by artifact/model name."""
	return {
		"logistic_regression": build_logistic_regression(random_state=random_state),
		"naive_bayes": build_naive_bayes(),
		"linear_svm": build_linear_svm(random_state=random_state),
		"calibrated_svm": build_calibrated_svm(random_state=random_state),
		"voting_classifier": build_voting_classifier(random_state=random_state),
	}


def train_model(model: Any, X_train: Any, y_train: pd.Series | list[str]) -> Any:
	"""Fit a model and return the fitted instance."""
	model.fit(X_train, y_train)
	return model


def evaluate_model(
	model: Any,
	X_eval: Any,
	y_eval: pd.Series | list[str],
	positive_label: str = "suicide",
) -> dict[str, float]:
	"""Compute common binary classification metrics for one model."""
	predictions = model.predict(X_eval)
	return {
		"accuracy": accuracy_score(y_eval, predictions),
		"precision": precision_score(y_eval, predictions, pos_label=positive_label),
		"recall": recall_score(y_eval, predictions, pos_label=positive_label),
		"f1": f1_score(y_eval, predictions, pos_label=positive_label),
	}


def train_and_evaluate_models(
	models: dict[str, Any],
	X_train: Any,
	y_train: pd.Series | list[str],
	X_val: Any,
	y_val: pd.Series | list[str],
	models_dir: str | Path | None = None,
	positive_label: str = "suicide",
) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Train all models, evaluate on validation split, and optionally persist models."""
	trained_models: dict[str, Any] = {}
	rows: list[dict[str, float | str]] = []

	output_dir: Path | None = None
	if models_dir is not None:
		output_dir = Path(models_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

	for model_name, model in models.items():
		fitted = train_model(model, X_train, y_train)
		metrics = evaluate_model(fitted, X_val, y_val, positive_label=positive_label)
		rows.append({"model": model_name, **metrics})
		trained_models[model_name] = fitted

		if output_dir is not None:
			save_model(fitted, output_dir / f"{model_name}.pkl")

	results_df = pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
	return results_df, trained_models


def save_model(model: Any, filepath: str | Path) -> None:
	"""Persist a fitted model to disk using pickle."""
	path = Path(filepath)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "wb") as file_handle:
		pickle.dump(model, file_handle)


def load_model(filepath: str | Path) -> Any:
	"""Load a model artifact from disk."""
	with open(filepath, "rb") as file_handle:
		return pickle.load(file_handle)


def get_model_params(model: Any) -> dict[str, Any]:
	"""Return estimator parameters for experiment reporting."""
	return model.get_params()
