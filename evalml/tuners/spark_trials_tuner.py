from skopt import Space

from evalml.tuners import NoParamsException, Tuner
from evalml.utils import deprecate_arg, get_random_state

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_absolute_error
from mlflow.models.signature import infer_signature


class SparkTrialsTuner(Tuner):
    """Random Search Optimizer.

    Example:
        >>> tuner = RandomSearchTuner({'My Component': {'param a': [0.0, 10.0], 'param b': ['a', 'b', 'c']}}, random_state=42)
        >>> proposal = tuner.propose()
        >>> assert proposal.keys() == {'My Component'}
        >>> assert proposal['My Component'] == {'param a': 3.7454011884736254, 'param b': 'c'}
    """

    def __init__(self, search_space={
                          'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
                          'learning_rate': hp.loguniform('learning_rate', -3, 0),
                          'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
                          'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
                          'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
                          'feature_fraction': hp.choice('feature_fraction', [0.5, 0.6, 0.7, 0.8, 0.9]),
                          'metric':'mape',
                          'objective': 'reg:squarederror',
                          'seed': 123, # Set a seed for deterministic training
                        }, 
                        parallelism=10, 
                        max_evals=30, 
                        random_state=None, 
                        random_seed=0, 
                        X_train=None,
                        X_test=None,
                        y_train=None,
                        y_test=None
                        ):
                                
        """ Sets up check for duplication if needed.

        Arguments:
            pipeline_hyperparameter_ranges (dict): a set of hyperparameter ranges corresponding to a pipeline's parameters
            random_state (int): Unused in this class. Defaults to 0.
            with_replacement (bool): If false, only unique hyperparameters will be shown
            replacement_max_attempts (int): The maximum number of tries to get a unique
                set of random parameters. Only used if tuner is initalized with
                with_replacement=True
        """
        self.search_space = search_space
        self.parallelism = parallelism
        self.max_evals = max_evals
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, params):
        # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
        mlflow.xgboost.autolog()

        with mlflow.start_run(nested=True):
            train = xgb.DMatrix(data=self.X_train, label=self.y_train)
            test = xgb.DMatrix(data=self.X_test, label=self.y_test)

            # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
            # is no longer improving.
            booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                                evals=[(test, "test")], early_stopping_rounds=50)

            y_predicted = booster.predict(test)

            mae_score = mean_absolute_error(self.y_test, y_predicted)
            max_error = sklearn.metrics.max_error(self.y_test, y_predicted)
            median_absolute_error = sklearn.metrics.median_absolute_error(self.y_test, y_predicted)
            explained_variance_score = sklearn.metrics.explained_variance_score(self.y_test, y_predicted)

            def mean_absolute_percentage_err(y_true, y_pred): 
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            mean_absolute_percentage_error = mean_absolute_percentage_err(self.y_test, y_predicted)

            mlflow.log_metric('mae', mae_score)
            mlflow.log_metric('mean_absolute_percentage_error', mean_absolute_percentage_error)
            mlflow.log_metric('max_error', max_error)
            mlflow.log_metric('median_absolute_error', median_absolute_error)
            mlflow.log_metric('explained_variance_score', explained_variance_score)

            signature = infer_signature(X_train.sample(10), booster.predict(xgb.DMatrix(self.X_train.sample(10))))
            mlflow.xgboost.log_model(booster, "model", signature=signature)

            # fmin minimizes the mean_absolute_percentage_error (mape)
            return {'status': STATUS_OK, 'loss': mean_absolute_percentage_error, 'booster': booster.attributes()}

    def start_run(self):
        # Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
        # A reasonable value for parallelism is the square root of max_evals.
        spark_trials = SparkTrials(parallelism=self.parallelism)

        # Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
        # run called "xgboost_models" .
        with mlflow.start_run(run_name='xgboost_models'):
            best_params = fmin(
                fn=self.train_model, 
                space=self.search_space, 
                algo=tpe.suggest, 
                max_evals=self.max_evals,
                trials=spark_trials, 
                rstate=np.random.RandomState(123)
                )
        return best_params

    
