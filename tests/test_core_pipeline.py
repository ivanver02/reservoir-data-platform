import numpy as np
import pandas as pd

from backend.modeling.evaluation import metrics


def test_metrics_report_basic_errors():
    actual = pd.Series([10.0, 20.0])
    predicted = pd.Series([11.0, 18.0])

    result = metrics(actual, predicted, capacity=100.0)

    assert result["mae"] == 1.5
    assert np.isclose(result["nmae_capacity"], 0.015)
