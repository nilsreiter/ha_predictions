"""Test for shape mismatch fix in Model.predict()."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the custom_components directory to path
ml_path = (
    Path(__file__).parent.parent.parent
    / "custom_components"
    / "ha_predictions"
    / "ml"
)
sys.path.insert(0, str(ml_path))

from model import Model  # noqa: E402
import logging  # noqa: E402


class TestModelPredictWithTargetColumn:
    """Test that Model.predict handles target column correctly."""

    def test_predict_with_target_column_present(self) -> None:
        """Test that predict works when target column is in input data.
        
        This tests the fix for: ValueError: shapes (1,5) and (2,) not aligned: 5 (dim 1) != 2 (dim 0)
        The issue occurred when the prediction input included the target column.
        """
        logger = logging.getLogger("test")
        model = Model(logger)

        # Training data with target column
        train_data = pd.DataFrame(
            {
                "feature1": ["a", "b", "a", "b", "a", "b"],
                "feature2": ["x", "y", "x", "y", "x", "y"],
                "target": ["class0", "class1", "class0", "class1", "class0", "class1"],
            }
        )

        # Train the model
        model.train_final(train_data)
        assert model.model_final is not None
        assert model.model_final.weights is not None
        # Model should be trained on 2 features (not 3)
        assert len(model.model_final.weights) == 2

        # Predict WITH target column (should work after fix)
        pred_data = pd.DataFrame(
            {
                "feature1": ["a"],
                "feature2": ["x"],
                "target": ["class0"],  # Target column is present
            }
        )

        # This should not raise ValueError
        result = model.predict(pred_data)
        assert result is not None
        predicted_class, probability = result
        assert predicted_class in ["class0", "class1"]
        assert 0 <= probability <= 1

    def test_predict_without_target_column(self) -> None:
        """Test that predict still works when target column is not in input data."""
        logger = logging.getLogger("test")
        model = Model(logger)

        # Training data
        train_data = pd.DataFrame(
            {
                "feature1": ["a", "b", "a", "b"],
                "feature2": ["x", "y", "x", "y"],
                "target": ["class0", "class1", "class0", "class1"],
            }
        )

        model.train_final(train_data)

        # Predict WITHOUT target column (traditional case)
        pred_data = pd.DataFrame({"feature1": ["b"], "feature2": ["y"]})

        result = model.predict(pred_data)
        assert result is not None
        predicted_class, probability = result
        assert predicted_class in ["class0", "class1"]
        assert 0 <= probability <= 1

    def test_predict_numeric_features_with_target(self) -> None:
        """Test with numeric features and target column present."""
        logger = logging.getLogger("test")
        model = Model(logger)

        # Training data with numeric features
        train_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [10, 20, 30, 40, 50, 60],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )

        model.train_final(train_data)
        assert len(model.model_final.weights) == 2

        # Predict with target column
        pred_data = pd.DataFrame(
            {"feature1": [2], "feature2": [20], "target": [1]}
        )

        result = model.predict(pred_data)
        assert result is not None
        predicted_class, probability = result
        assert predicted_class in [0, 1]
        assert 0 <= probability <= 1

    def test_predict_mixed_features_with_target(self) -> None:
        """Test with mixed categorical and numeric features."""
        logger = logging.getLogger("test")
        model = Model(logger)

        train_data = pd.DataFrame(
            {
                "cat_feature": ["a", "b", "a", "b", "a", "b"],
                "num_feature": [1, 2, 1, 2, 1, 2],
                "target": ["yes", "no", "yes", "no", "yes", "no"],
            }
        )

        model.train_final(train_data)
        assert len(model.model_final.weights) == 2

        # Predict with all columns including target
        pred_data = pd.DataFrame(
            {"cat_feature": ["a"], "num_feature": [1], "target": ["yes"]}
        )

        result = model.predict(pred_data)
        assert result is not None
        predicted_class, probability = result
        assert predicted_class in ["yes", "no"]
        assert 0 <= probability <= 1
