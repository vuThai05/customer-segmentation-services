from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "app.py"
MODULE_SPEC = spec_from_file_location("customer_segmentation_app", MODULE_PATH)
APP_MODULE = module_from_spec(MODULE_SPEC)
sys.modules["customer_segmentation_app"] = APP_MODULE
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(APP_MODULE)

build_export_df = APP_MODULE.build_export_df
prepare_numeric_features = APP_MODULE.prepare_numeric_features
run_segmentation = APP_MODULE.run_segmentation


def test_prepare_numeric_features_filters_numeric_and_imputes_median() -> None:
    df = pd.DataFrame(
        {
            "age": [20.0, np.nan, 40.0],
            "income": [100.0, 150.0, np.nan],
            "segment": ["A", "B", "C"],
        }
    )

    prepared = prepare_numeric_features(df)

    assert prepared.numeric_columns == ["age", "income"]
    assert prepared.missing_values_fixed == 2
    assert prepared.numeric_df.loc[1, "age"] == 30.0
    assert prepared.numeric_df.loc[2, "income"] == 125.0
    assert prepared.scaled_array.shape == (3, 2)


def test_prepare_numeric_features_handles_no_numeric_columns() -> None:
    df = pd.DataFrame({"segment": ["A", "B"], "tier": ["Gold", "Silver"]})

    prepared = prepare_numeric_features(df)

    assert prepared.numeric_columns == []
    assert prepared.missing_values_fixed == 0
    assert prepared.numeric_df.empty
    assert prepared.scaled_array.shape == (2, 0)
    assert prepared.imputer is None
    assert prepared.scaler is None


def test_run_segmentation_returns_labels_and_skips_pca_for_one_numeric_column() -> None:
    df = pd.DataFrame({"spend": [10.0, 12.0, 50.0, 52.0]})
    prepared = prepare_numeric_features(df)

    result = run_segmentation(prepared.scaled_array, n_groups=2)

    assert len(result.labels) == len(df)
    assert sorted(np.unique(result.labels).tolist()) == [0, 1]
    assert result.pca_model is None
    assert result.pca_coordinates is None
    assert result.representative_coordinates is None


def test_build_export_df_appends_cluster_label() -> None:
    df = pd.DataFrame({"customer_id": [101, 102, 103], "spend": [20, 25, 100]})
    labels = np.array([1, 1, 0])

    export_df = build_export_df(df, labels)

    assert list(export_df.columns) == ["customer_id", "spend", "Cluster_Label"]
    assert export_df["Cluster_Label"].tolist() == ["Group 2", "Group 2", "Group 1"]
