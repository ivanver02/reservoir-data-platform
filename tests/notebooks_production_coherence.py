from pathlib import Path
import pandas as pd
import pytest
import sys

sys.path.append(str(Path.cwd()))
from backend.config.settings import PATHS


def _assert_same(df_notebook: pd.DataFrame, df_prod: pd.DataFrame):
	assert df_notebook.shape == df_prod.shape, (
		f"Shape mismatch notebook={df_notebook.shape} prod={df_prod.shape}"
	)
	comparison = (df_prod == df_notebook)
	assert ((comparison.sum() == df_prod.notna().sum()).all()), (
		"Dataframes differ (after aligning by index/columns). "
		"Consider inspecting differing cells: "
		f"mismatches={(~comparison & df_prod.notna()).sum().sum()}"
	)


def test_water_cleaned_coherence():
	nb = pd.read_parquet(PATHS['cleaned_data_notebooks'] / 'water_cleaned.parquet')
	prod = pd.read_csv(PATHS['cleaned_data_production'] / 'water_cleaned.csv')
	_assert_same(nb, prod)


def test_reservoirs_cleaned_coherence():
	nb = pd.read_csv(PATHS['cleaned_data_notebooks'] / 'reservoirs_cleaned.csv')
	prod = pd.read_csv(PATHS['cleaned_data_production'] / 'reservoirs_cleaned.csv')
	_assert_same(nb, prod)


def test_detailed_reservoirs_cleaned_coherence():
	nb = pd.read_csv(PATHS['cleaned_data_notebooks'] / 'detailed_reservoirs_cleaned.csv')
	prod = pd.read_csv(PATHS['cleaned_data_production'] / 'detailed_reservoirs_cleaned.csv')
	_assert_same(nb, prod)


def test_water_definitive_coherence():
	nb = pd.read_parquet(PATHS['definitive_notebooks'] / 'water_definitive.parquet')
	prod = pd.read_csv(PATHS['definitive_production'] / 'water_definitive.csv')
	_assert_same(nb, prod)


def test_reservoirs_definitive_coherence():
	nb = pd.read_parquet(PATHS['definitive_notebooks'] / 'reservoirs_definitive.parquet')
	prod = pd.read_csv(PATHS['definitive_production'] / 'reservoirs_definitive.csv')
	_assert_same(nb, prod)


def test_detailed_reservoirs_definitive_coherence():
	nb = pd.read_parquet(PATHS['definitive_notebooks'] / 'detailed_reservoirs_definitive.parquet')
	prod = pd.read_csv(PATHS['definitive_production'] / 'detailed_reservoirs_definitive.csv')
	_assert_same(nb, prod)


def test_reservoirs_merged_definitive_coherence():
	nb = pd.read_parquet(PATHS['definitive_notebooks'] / 'reservoirs_merged.parquet')
	prod = pd.read_csv(PATHS['definitive_production'] / 'reservoirs_merged_definitive.csv')
	_assert_same(nb, prod)


if __name__ == '__main__':
	sys.exit(_pytest.main([__file__]))

