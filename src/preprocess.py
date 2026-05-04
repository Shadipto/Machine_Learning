from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_TEXT_COLUMN = "text"
DEFAULT_TARGET_COLUMN = "class"
DEFAULT_CLEANED_COLUMN = "cleaned_text"
DEFAULT_DROP_COLUMNS = ("Unnamed: 0",)
DEFAULT_RANDOM_SEED = 42


def load_dataset(filepath: str | Path) -> pd.DataFrame:
	"""Load a CSV dataset into a pandas DataFrame.

	Args:
		filepath: Absolute or relative CSV path.

	Returns:
		Loaded DataFrame.
	"""
	return pd.read_csv(filepath)


def clean_text(text: object) -> str:
	"""Normalize raw text with deterministic rules used in the notebook pipeline.

	Rules applied:
	1. Lowercase
	2. Remove URLs
	3. Remove HTML tags
	4. Remove punctuation
	5. Remove digits
	6. Collapse whitespace

	Args:
		text: Raw text-like object.

	Returns:
		Cleaned text string.
	"""
	cleaned = str(text).lower()
	cleaned = re.sub(r"http\S+|www\S+", " ", cleaned)
	cleaned = re.sub(r"<.*?>", " ", cleaned)
	cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
	cleaned = re.sub(r"\d+", " ", cleaned)
	cleaned = re.sub(r"\s+", " ", cleaned).strip()
	return cleaned


def clean_dataset(
	df: pd.DataFrame,
	text_column: str = DEFAULT_TEXT_COLUMN,
	target_column: str = DEFAULT_TARGET_COLUMN,
	cleaned_column: str = DEFAULT_CLEANED_COLUMN,
	drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> pd.DataFrame:
	"""Create a cleaned dataset with a dedicated cleaned-text column.

	Args:
		df: Raw input dataset.
		text_column: Name of input text column.
		target_column: Name of target label column.
		cleaned_column: Name of output cleaned text column.
		drop_columns: Optional columns to drop if present.

	Returns:
		New DataFrame containing the target and cleaned text.

	Raises:
		ValueError: If required columns are missing.
	"""
	required = {text_column, target_column}
	missing = sorted(required - set(df.columns))
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	cleaned_df = df.copy()
	existing_drop_cols = [col for col in drop_columns if col in cleaned_df.columns]
	if existing_drop_cols:
		cleaned_df = cleaned_df.drop(columns=existing_drop_cols)

	cleaned_df[text_column] = cleaned_df[text_column].fillna("").astype(str)
	cleaned_df[cleaned_column] = cleaned_df[text_column].apply(clean_text)
	return cleaned_df


def split_dataset(
	df: pd.DataFrame,
	target_column: str = DEFAULT_TARGET_COLUMN,
	train_size: float = 0.8,
	val_size: float = 0.1,
	test_size: float = 0.1,
	random_state: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split a DataFrame into stratified train/validation/test partitions.

	Args:
		df: Input DataFrame.
		target_column: Label column used for stratification.
		train_size: Proportion of training split.
		val_size: Proportion of validation split.
		test_size: Proportion of test split.
		random_state: Seed for reproducibility.

	Returns:
		Tuple of (train_df, val_df, test_df).

	Raises:
		ValueError: If split sizes are invalid or target is missing.
	"""
	if target_column not in df.columns:
		raise ValueError(f"Target column not found: {target_column}")

	total = train_size + val_size + test_size
	if abs(total - 1.0) > 1e-8:
		raise ValueError("train_size + val_size + test_size must equal 1.0")

	train_val_df, test_df = train_test_split(
		df,
		test_size=test_size,
		random_state=random_state,
		stratify=df[target_column],
	)

	val_ratio_within_train_val = val_size / (train_size + val_size)
	train_df, val_df = train_test_split(
		train_val_df,
		test_size=val_ratio_within_train_val,
		random_state=random_state,
		stratify=train_val_df[target_column],
	)

	return (
		train_df.reset_index(drop=True),
		val_df.reset_index(drop=True),
		test_df.reset_index(drop=True),
	)
