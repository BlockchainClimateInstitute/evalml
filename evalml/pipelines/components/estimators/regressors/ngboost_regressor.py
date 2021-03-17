from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _rename_column_names_to_numeric,
    deprecate_arg,
    import_or_raise
)


from scipy.stats import norm, lognorm, gstd
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor

import ngboost , time
from ngboost import NGBRegressor
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal, LogNormal, Exponential
from ngboost.scores import MLE, CRPS, LogScore


class NGBoostRegressor(Estimator):
    """NGBoost Regressor."""
    name = "NGBoost Regressor"
    
    hyperparameter_ranges = {}
    
    model_family = ModelFamily.NGBOOST
    supported_problem_types = [ProblemTypes.REGRESSION]

    # ngboost supports seeds, these limits ensure the random seed generated below
    # is within that range.
    SEED_MIN = -2**31
    SEED_MAX = 2**31 - 1

    def __init__(self, 
                 conf = .95,
                 confs=None,
                 criterion="mse",
                 max_features="auto",
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 eta=0.1, 
                 max_depth=6, 
                 min_child_weight=1, 
                 n_estimators=5000, 
                 random_state=0, 
                 **kwargs):
        
        parameters = {"criterion": criterion,
                      "max_features": max_features,
                      "max_depth": max_depth,
                      "min_samples_split": min_samples_split,
                      "min_weight_fraction_leaf": min_weight_fraction_leaf}
        
        parameters.update(kwargs)
        dt_regressor = SKDecisionTreeRegressor(random_state=random_state, **parameters)
        ngb_error_msg = "NGBoost is not installed. Please install using `pip install ngboost.`"
#         ngb = import_or_raise("ngboost", error_msg=ngb_error_msg)
        ngb_regressor = NGBRegressor(
                           Base=dt_regressor, 
                           Dist=LogNormal, 
                           Score=MLE,
                           n_estimators=n_estimators,
                           verbose=True
                          )
        self.conf = conf
        self.confs = confs
        super().__init__(parameters=parameters,
                         component_obj=ngb_regressor,
                         random_state=random_state)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame): 
            X = pd.DataFrame(X)
            
        X_t = X
        X_t['prediction'] = np.where(X_t['prediction'] <= 0.0, 1.0, X_t['prediction'])
        X_trains, X_holdouts, y_trains, y_holdouts = train_test_split(X_t, y.fillna(10000.0), test_size=0.2, random_state=0)
        return self._component_obj.fit(X_trains, y_trains, X_val=X_holdouts, Y_val=y_holdouts, early_stopping_rounds=20)
    
    def predict(self, X):
        if not isinstance(X, pd.DataFrame): 
            X = pd.DataFrame(X)
            
        X_t = X
        self.preds=[]
        X_t['prediction'] = np.where(X_t['prediction'] <= 0.0, 1.0, X_t['prediction'])
        preds = self.dynamic_pred_dist(X_t)
        return preds
    
    def predict_dist(self, X, conf=0.95):
        ts = time.time() 
        resp = pd.DataFrame({})
        if not isinstance(X, pd.DataFrame): 
            X = pd.DataFrame(X)
            
        X_t = X
        preds = X_t['prediction']
        dist = self._component_obj.pred_dist(X_t)
        s = dist.params.get('s')
        scale = dist.params.get('scale')
        intervals = lognorm(s=s, scale=scale).interval(conf)
        stdev = lognorm.std(s, loc=0, scale=scale)
        lower_bound, upper_bound = intervals[0], intervals[1]
        resp['avm_val'] = [int(round(x,0)) for x in preds]
        resp['avm_lower'] = [int(round(x,0)) for x in lower_bound]
        resp['avm_upper'] = [int(round(x,0)) for x in upper_bound]
        resp['conf'] = [conf for x in range(len(resp))]
        resp['stdev'] = [int(round(x,0)) for x in stdev]
        resp['timestamp'] = [str(ts) for x in range(len(resp))]
        resp['avm_lower'] = np.where(resp['avm_lower'] < 0, resp['avm_val'] - resp['avm_val']*.2, resp['avm_lower'])
        resp['avm_lower'] = [int(round(x,0)) for x in resp['avm_lower'].values]
        resp['avm_upper'] = [int(round(x,0)) for x in resp['avm_upper'].values]
        return resp
    
    def dynamic_pred_dist(self, X, conf=0.95):    
        if not isinstance(X, pd.DataFrame): 
            X = pd.DataFrame(X)
            
        l = []
        for ct in range(len(X)):
            index = X.reset_index()['index'].values[ct]
            row = X.reset_index()[X.reset_index()['index']==index]
            keep_trying = True
            for conf in self.confs:
                if keep_trying:
                    x = self.predict_dist(row, conf=conf)
                    if conf < .3:
                        l.append(x)
                        keep_trying = False
                    elif x['avm_upper'].astype(float).values[0] - x['avm_lower'].astype(float).values[0] < 0.2 * x['avm_val'].astype(float).values[0]:
                        l.append(x)
                        keep_trying = False
             
        X_t = pd.concat(l)
        return X_t
    
    @property
    def feature_importance(self, ):
        return self.preprocessing_pipeline.feature_importance



