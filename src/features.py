from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_MAX_FEATURES = 20000
DEFAULT_STOP_WORDS = "english"
DEFAULT_NGRAM_RANGE = (1, 1)


def _normalize_text_series(texts: pd.Series | list[str]) -> pd.Series:
	"""Return text input as normalized pandas Series for vectorization."""
	if isinstance(texts, pd.Series):
		series = texts
	else:
		series = pd.Series(texts)
	return series.fillna("").astype(str)


def build_tfidf_vectorizer(
	max_features: int = DEFAULT_MAX_FEATURES,
	stop_words: str | None = DEFAULT_STOP_WORDS,
	ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> TfidfVectorizer:
	"""Create a TF-IDF vectorizer with project defaults."""
	return TfidfVectorizer(
		stop_words=stop_words,
		max_features=max_features,
		ngram_range=ngram_range,
	)


def fit_vectorizer(
	train_texts: pd.Series | list[str],
	vectorizer: TfidfVectorizer | None = None,
) -> tuple[TfidfVectorizer, sparse.spmatrix]:
	"""Fit TF-IDF vectorizer on training texts and return transformed matrix."""
	vec = vectorizer or build_tfidf_vectorizer()
	normalized = _normalize_text_series(train_texts)
	x_train = vec.fit_transform(normalized)
	return vec, x_train


def transform_texts(
	texts: pd.Series | list[str],
	vectorizer: TfidfVectorizer,
) -> sparse.spmatrix:
	"""Transform texts using an already fitted TF-IDF vectorizer."""
	normalized = _normalize_text_series(texts)
	return vectorizer.transform(normalized)


def fit_transform_splits(
	train_texts: pd.Series | list[str],
	val_texts: pd.Series | list[str],
	test_texts: pd.Series | list[str],
	vectorizer: TfidfVectorizer | None = None,
) -> tuple[TfidfVectorizer, sparse.spmatrix, sparse.spmatrix, sparse.spmatrix]:
	"""Fit on train split and transform validation/test splits with same vectorizer."""
	fitted_vec, x_train = fit_vectorizer(train_texts, vectorizer=vectorizer)
	x_val = transform_texts(val_texts, fitted_vec)
	x_test = transform_texts(test_texts, fitted_vec)
	return fitted_vec, x_train, x_val, x_test


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str | Path) -> None:
	"""Persist a fitted TF-IDF vectorizer to disk with pickle."""
	path = Path(filepath)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "wb") as file_handle:
		pickle.dump(vectorizer, file_handle)


def load_vectorizer(filepath: str | Path) -> TfidfVectorizer:
	"""Load a previously saved TF-IDF vectorizer from disk."""
	with open(filepath, "rb") as file_handle:
		return pickle.load(file_handle)


def get_feature_names(vectorizer: TfidfVectorizer) -> list[str]:
	"""Return ordered TF-IDF feature names from the fitted vectorizer."""
	return vectorizer.get_feature_names_out().tolist()


def get_vectorizer_params(vectorizer: TfidfVectorizer) -> dict[str, Any]:
	"""Expose key vectorizer parameters for reporting and reproducibility."""
	return vectorizer.get_params()
