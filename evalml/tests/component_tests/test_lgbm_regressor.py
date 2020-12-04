from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines import LightGBMRegressor
from evalml.problem_types import ProblemTypes
from evalml.utils import SEED_BOUNDS

lgbm = importorskip('lightgbm', reason='Skipping test because lightgbm not installed')


def test_model_family():
    assert LightGBMRegressor.model_family == ModelFamily.LIGHTGBM


def test_problem_types():
    assert set(LightGBMRegressor.supported_problem_types) == {ProblemTypes.REGRESSION}


def test_lightgbm_regressor_random_state_bounds_seed(X_y_regression):
    """ensure lightgbm's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_regression
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = LightGBMRegressor(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.min_bound)
    clf.fit(X, y)
    clf = LightGBMRegressor(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.max_bound)
    clf.fit(X, y)


def test_lightgbm_regressor_random_state_bounds_rng(X_y_regression):
    """when a RNG is inputted for random_state, ensure the sample we take to get a random seed for lightgbm is in lightgbm's supported range"""

    def make_mock_random_state(return_value):

        class MockRandomState(np.random.RandomState):

            def randint(self, min_bound, max_bound):
                return return_value
        return MockRandomState()

    X, y = X_y_regression
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    rng = make_mock_random_state(LightGBMRegressor.SEED_MIN)
    clf = LightGBMRegressor(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)
    rng = make_mock_random_state(LightGBMRegressor.SEED_MAX)
    clf = LightGBMRegressor(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)


def test_fit_predict_regression(X_y_regression):
    X, y = X_y_regression

    sk_clf = lgbm.sklearn.LGBMRegressor(random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)

    clf = LightGBMRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=3)


def test_feature_importance(X_y_regression):
    X, y = X_y_regression

    clf = LightGBMRegressor()
    sk_clf = lgbm.sklearn.LGBMRegressor(random_state=0)
    sk_clf.fit(X, y)
    sk_feature_importance = sk_clf.feature_importances_

    clf.fit(X, y)
    feature_importance = clf.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=3)


def test_fit_string_features(X_y_regression):
    X, y = X_y_regression
    X = pd.DataFrame(X)
    X['string_col'] = 'abc'

    # lightGBM requires input args to be int, float, or bool, not string
    X_expected = X.copy()
    X_expected['string_col'] = 0.0

    clf = lgbm.sklearn.LGBMRegressor(random_state=0)
    clf.fit(X_expected, y, categorical_feature=['string_col'])
    y_pred_sk = clf.predict(X_expected)

    clf = LightGBMRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=3)


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_correct_args(mock_fit, mock_predict, X_y_regression):
    X, y = X_y_regression
    X = pd.DataFrame(X)

    # add object (string) and categorical data.
    X['string_col'] = 'abc'
    X['string_col'].iloc[len(X) // 2:] = 'cba'
    X['categorical_data'] = 'square'
    X['categorical_data'].iloc[len(X) // 2:] = 'circle'
    X['categorical_data'] = X['categorical_data'].astype('category')

    # create the expected result, which is a dataframe with int values in the categorical column and dtype=category
    X_expected = X.copy()
    X_expected = X_expected.replace(["abc", "cba"], [0.0, 1.0])
    X_expected = X_expected.replace(["square", "circle"], [1.0, 0.0])
    X_expected[['string_col', 'categorical_data']] = X_expected[['string_col', 'categorical_data']].astype('category')

    # rename the columns to be the indices
    X_expected.columns = np.arange(X_expected.shape[1])

    clf = LightGBMRegressor()
    clf.fit(X, y)
    arg_X = mock_fit.call_args[0][0]
    assert_frame_equal(X_expected, arg_X)

    clf.predict(X)
    arg_X = mock_predict.call_args[0][0]
    assert_frame_equal(X_expected, arg_X)


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_categorical_data_subset(mock_fit, mock_predict, X_y_regression):
    X = pd.DataFrame({"feature_1": [0, 0, 1, 1, 0, 1], "feature_2": ["a", "a", "b", "b", "c", "c"]})
    y = pd.Series([1, 1, 0, 0, 0, 1])
    X_expected = pd.DataFrame({0: [0, 0, 1, 1, 0, 1], 1: [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]})
    X_expected.iloc[:, 1] = X_expected.iloc[:, 1].astype('category')

    X_subset = pd.DataFrame({"feature_1": [1, 0], "feature_2": ["c", "a"]})
    X_expected_subset = pd.DataFrame({0: [1, 0], 1: [2.0, 0.0]})
    X_expected_subset.iloc[:, 1] = X_expected_subset.iloc[:, 1].astype('category')

    clf = LightGBMRegressor()
    clf.fit(X, y)
    arg_X = mock_fit.call_args[0][0]
    assert_frame_equal(X_expected, arg_X)

    # determine whether predict and predict_proba perform as expected with the subset of categorical data
    clf.predict(X_subset)
    arg_X = mock_predict.call_args[0][0]
    assert_frame_equal(X_expected_subset, arg_X)


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_multiple_fit(mock_fit, mock_predict):
    y = pd.Series([1] * 4)
    X1_fit = pd.DataFrame({"feature": ["a", "b", "c", "c"]})
    X1_fit_expected = pd.DataFrame({0: [0.0, 1.0, 2.0, 2.0]}, dtype='category')
    X1_predict = pd.DataFrame({"feature": ["a", "a", "b", "c"]})
    X1_predict_expected = pd.DataFrame({0: [0.0, 0.0, 1.0, 2.0]}, dtype='category')

    clf = LightGBMRegressor()
    clf.fit(X1_fit, y)
    assert_frame_equal(X1_fit_expected, mock_fit.call_args[0][0])
    clf.predict(X1_predict)
    assert_frame_equal(X1_predict_expected, mock_predict.call_args[0][0])

    # Check if it will fit a different dataset with new variable
    X2_fit = pd.DataFrame({"feature": ["c", "b", "a", "d"]})
    X2_fit_expected = pd.DataFrame({0: [2.0, 1.0, 0.0, 3.0]}, dtype='category')
    X2_predict = pd.DataFrame({"feature": ["d", "c", "b", "a"]})
    X2_predict_expected = pd.DataFrame({0: [3.0, 2.0, 1.0, 0.0]}, dtype='category')

    clf = LightGBMRegressor()
    clf.fit(X2_fit, y)
    assert_frame_equal(X2_fit_expected, mock_fit.call_args[0][0])
    clf.predict(X2_predict)
    assert_frame_equal(X2_predict_expected, mock_predict.call_args[0][0])


def test_regression_rf(X_y_regression):
    X, y = X_y_regression

    with pytest.raises(lgbm.basic.LightGBMError, match="bagging_fraction"):
        clf = LightGBMRegressor(boosting_type="rf", bagging_freq=1, bagging_fraction=1.01)
        clf.fit(X, y)

    clf = LightGBMRegressor(boosting_type="rf", bagging_freq=0)
    clf.fit(X, y)
    assert clf.parameters['bagging_freq'] == 0
    assert clf.parameters['bagging_fraction'] == 0.9


def test_regression_goss(X_y_regression):
    X, y = X_y_regression
    clf = LightGBMRegressor(boosting_type="goss")
    clf.fit(X, y)
    assert clf.parameters['bagging_freq'] == 0
    assert clf.parameters['bagging_fraction'] == 0.9
