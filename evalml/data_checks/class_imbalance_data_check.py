
from evalml.data_checks import (
    DataCheck,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types


class ClassImbalanceDataCheck(DataCheck):
    """Checks if any target labels are imbalanced beyond a threshold. Use for classification problems"""

    def __init__(self, threshold=0.1, min_samples=100, num_cv_folds=3):
        """Check if any of the target labels are imbalanced, or if the number of values for each target
           are below 2 times the number of cv folds

        Arguments:
            threshold (float): The minimum threshold allowed for class imbalance before a warning is raised.
                A perfectly balanced dataset would have a threshold of (1/n_classes), ie 0.50 for binary classes.
                Defaults to 0.10
            min_samples (int): The minimum number of samples per accepted class. If the minority class is both below the threshold and min_samples,
                then we consider this severely imbalanced. Must be greater than 0. Defaults to 100.
            num_cv_folds (int): The number of cross-validation folds. Must be positive. Choose 0 to ignore this warning.
        """
        if threshold <= 0 or threshold > 0.5:
            raise ValueError("Provided threshold {} is not within the range (0, 0.5]".format(threshold))
        self.threshold = threshold
        if min_samples <= 0:
            raise ValueError("Provided value min_samples {} is not greater than 0".format(min_samples))
        self.min_samples = min_samples
        if num_cv_folds < 0:
            raise ValueError("Provided number of CV folds {} is less than 0".format(num_cv_folds))
        self.cv_folds = num_cv_folds * 2

    def validate(self, X, y):
        """Checks if any target labels are imbalanced beyond a threshold for binary and multiclass problems
            Ignores NaN values in target labels if they appear.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features. Ignored.
            y (ww.DataColumn, pd.Series, np.ndarray): Target labels to check for imbalanced data.

        Returns:
            dict: Dictionary with DataCheckWarnings if imbalance in classes is less than the threshold,
                  and DataCheckErrors if the number of values for each target is below 2 * num_cv_folds.

        Example:
            >>> import pandas as pd
            >>> X = pd.DataFrame()
            >>> y = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            >>> target_check = ClassImbalanceDataCheck(threshold=0.10)
            >>> assert target_check.validate(X, y) == {"errors": [{"message": "The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0]",\
                                                                   "data_check_name": "ClassImbalanceDataCheck",\
                                                                   "level": "error",\
                                                                   "code": "CLASS_IMBALANCE_BELOW_FOLDS",\
                                                                   "details": {"target_values": [0]}}],\
                                                     "warnings": [{"message": "The following labels fall below 10% of the target: [0]",\
                                                                   "data_check_name": "ClassImbalanceDataCheck",\
                                                                   "level": "warning",\
                                                                   "code": "CLASS_IMBALANCE_BELOW_THRESHOLD",\
                                                                   "details": {"target_values": [0]}},\
                                                                   {"message": "The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than 100 samples: [0]",\
                                                                   "data_check_name": "ClassImbalanceDataCheck",\
                                                                   "level": "warning",\
                                                                   "code": "CLASS_IMBALANCE_SEVERE",\
                                                                   "details": {"target_values": [0]}}],\
                                                     "actions": []}
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series())

        fold_counts = y.value_counts(normalize=False)
        # search for targets that occur less than twice the number of cv folds first
        below_threshold_folds = fold_counts.where(fold_counts < self.cv_folds).dropna()
        if len(below_threshold_folds):
            below_threshold_values = below_threshold_folds.index.tolist()
            error_msg = "The number of instances of these targets is less than 2 * the number of cross folds = {} instances: {}"
            DataCheck._add_message(DataCheckError(message=error_msg.format(self.cv_folds, below_threshold_values),
                                                  data_check_name=self.name,
                                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                                  details={"target_values": below_threshold_values}), results)

        counts = fold_counts / fold_counts.sum()
        below_threshold = counts.where(counts < self.threshold).dropna()
        # if there are items that occur less than the threshold, add them to the list of results
        if len(below_threshold):
            below_threshold_values = below_threshold.index.tolist()
            warning_msg = "The following labels fall below {:.0f}% of the target: {}"
            DataCheck._add_message(DataCheckWarning(message=warning_msg.format(self.threshold * 100, below_threshold_values),
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                                    details={"target_values": below_threshold_values}), results)
        sample_counts = fold_counts.where(fold_counts < self.min_samples).dropna()
        if len(below_threshold) and len(sample_counts):
            sample_count_values = sample_counts.index.tolist()
            severe_imbalance = [v for v in sample_count_values if v in below_threshold]
            warning_msg = "The following labels in the target have severe class imbalance because they fall under {:.0f}% of the target and have less than {} samples: {}"
            DataCheck._add_message(DataCheckWarning(message=warning_msg.format(self.threshold * 100, self.min_samples, severe_imbalance),
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                                                    details={"target_values": severe_imbalance}), results)
        return results
