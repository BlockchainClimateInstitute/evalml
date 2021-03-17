
from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types





class RegressionPipeline(PipelineBase):
    """Pipeline subclass for all regression pipelines."""
    problem_type = ProblemTypes.REGRESSION

    def fit(self, X, y):
        """Build a regression model.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training data of length [n_samples]

        Returns:
            self

        """
        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
        y = _convert_woodwork_types_wrapper(y.to_series())
        if y.dtype not in numeric_dtypes:
            raise ValueError(f"Regression pipeline cannot handle targets with dtype: {y.dtype}")
        self._fit(X, y)
        return self

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, or np.ndarray): True values of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        y_predicted = self.predict(X)
        return self._score_all_objectives(X, y, y_predicted, y_pred_proba=None, objectives=objectives)

    def custom_score(self, X, y):
        from sklearn import metrics
        
        if not isinstance(X, pd.DataFrame): 
            X = pd.DataFrame(X)
            
        preds = self.predict(X)
        y_predicted = preds['avm_val']
        y_lower = preds['avm_lower']
        y_upper = preds['avm_upper']
        confs = preds['conf']
            
        def mean_squared_error(y_true, y_predicted, X=None):
            return metrics.mean_squared_error(y_true, y_predicted, squared=False)
        
        def r2_score(y_true, y_predicted, X=None):
            return metrics.r2_score(y_true, y_predicted)
        
        def mean_absolute_error(y_true, y_predicted, X=None):
            return metrics.mean_absolute_error(y_true, y_predicted)
        
        def median_absolute_error(y_true, y_predicted, X=None):
            return metrics.median_absolute_error(y_true, y_predicted)
        
        def max_error(y_true, y_predicted, X=None):
            return metrics.max_error(y_true, y_predicted)

        def mean_absolute_percentage_err(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def accuracy_in_range(y_true, y_lower, y_upper): 
            benchmark = [1.0 for x in range(len(y_true))]
            return metrics.accuracy(benchmark, np.where(np.array(y_lower) <= np.array(y_true) <= np.array(y_lower), 1.0, 0.0))
        
        dic = {
            'accuracy_in_range':[mean_squared_error(y, y_lower, y_upper)],
            'mean_squared_error':[mean_squared_error(y, y_predicted)],
            'r2_score':[r2_score(y, y_predicted)],
            'mean_absolute_error':[mean_absolute_error(y, y_predicted)],
            'mean_absolute_percentage_error':[mean_absolute_percentage_err(y, y_predicted)],
            'median_absolute_error':[median_absolute_error(y, y_predicted)],
            'max_error':[max_error(y, y_predicted)],
        }
        return pd.DataFrame(dic)
    
    def _plot_accuracy_by_price_range(self, y_holdouts, y_predicted):
        model_name='avm'

        df_acc = pd.DataFrame({})
        df_acc['y_test'] = list(y_holdouts)
        df_acc['preds'] = list(y_predicted.astype(int))
        df_acc['error'] = abs((df_acc['preds'] - df_acc['y_test'])/df_acc['y_test'])
        df_acc['percent_error'] = (df_acc['preds'] - df_acc['y_test'])/df_acc['y_test']


        ctt = []
        lim=5
        try: 
            if len(df_acc[df_acc['y_test']<=50000]) > 5: ctt.append('df_0')
        except: pass
        try: 
            if len(df_acc[(df_acc['y_test']>=50000) & (df_acc['y_test']<=100000)]) > lim: ctt.append('df_5')
        except: pass
        try: 
            if len(df_acc[(df_acc['y_test']>=100000) & (df_acc['y_test']<=150000)]) > lim: ctt.append('df_10')
        except: pass
        try: 
            if len(df_acc[(df_acc['y_test']>=150000) & (df_acc['y_test']<=200000)]) > lim: ctt.append('df_15')
        except: pass
        try: 
            if len(df_acc[(df_acc['y_test']>=200000) & (df_acc['y_test']<=250000)]) > lim: ctt.append('df_20')
        except: pass
        try: 
            if len(df_acc[(df_acc['y_test']>=250000) & (df_acc['y_test']<=300000)]) > lim: ctt.append('df_25')
        except: pass
        try: 
            if len(df_acc[(df_acc['y_test']>=300000) & (df_acc['y_test']<=350000)]) > lim: ctt.append('df_30')
        except: pass
        try: 
            if len(df_acc[(df_acc['y_test']>=350000) & (df_acc['y_test']<=400000)]) > lim: ctt.append('df_35')
        except: pass
        try: 
            if len(df_acc[df_acc['y_test']>=400000]) > lim: ctt.append('df_40')
        except: pass

        def troubleshoot(ctt, df_acc):

            acc_2 = len(ctt)*[0]
            acc_5 = len(ctt)*[0]
            acc_10 = len(ctt)*[0]
            acc_20 = len(ctt)*[0]

            x = 0
            if 'df_0' in ctt:
                try:
                    df_0 = df_acc[df_acc['y_test']<=50000]
                    acc_2[x] = len(df_0[df_0['error']<0.02])/len(df_0)
                    acc_5[x] = len(df_0[df_0['error']<0.05])/len(df_0)
                    acc_10[x] = len(df_0[df_0['error']<0.1])/len(df_0)
                    acc_20[x] = len(df_0[df_0['error']<0.2])/len(df_0)
                    x += 1
                except: 
                    ctt.pop('df_0')
                    troubleshoot(ctt, df_acc)

            if 'df_5' in ctt:
                try:
                    df_5 = df_acc[(df_acc['y_test']>=50000) & (df_acc['y_test']<=100000)]
                    acc_2[x] = len(df_5[df_5['error']<0.02])/len(df_5)
                    acc_5[x] = len(df_5[df_5['error']<0.05])/len(df_5)
                    acc_10[x] = len(df_5[df_5['error']<0.10])/len(df_5)
                    acc_20[x] = len(df_5[df_5['error']<0.2])/len(df_5)
                    x += 1
                except: 
                    ctt.pop('df_5')
                    troubleshoot(ctt, df_acc)

            if 'df_10' in ctt:
                try:
                    df_10 = df_acc[(df_acc['y_test']>=100000) & (df_acc['y_test']<=150000)]
                    acc_2[x] = len(df_10[df_10['error']<0.02])/len(df_10)
                    acc_5[x] = len(df_10[df_10['error']<0.05])/len(df_10)
                    acc_10[x] = len(df_10[df_10['error']<0.10])/len(df_10)
                    acc_20[x] = len(df_10[df_10['error']<0.2])/len(df_10)
                    x += 1
                except: 
                    ctt.pop('df_10')
                    troubleshoot(ctt, df_acc)

            if 'df_10' in ctt:
                try:
                    df_15 = df_acc[(df_acc['y_test']>=150000) & (df_acc['y_test']<=200000)]
                    acc_2[x] = len(df_15[df_15['error']<0.02])/len(df_15)
                    acc_5[x] = len(df_15[df_15['error']<0.05])/len(df_15)
                    acc_10[x] = len(df_15[df_15['error']<0.1])/len(df_15)
                    acc_20[x] = len(df_15[df_15['error']<0.2])/len(df_15)
                    x += 1
                except: 
                    ctt.pop('df_10')
                    troubleshoot(ctt, df_acc)

            if 'df_20' in ctt:
                try:
                    df_20 = df_acc[(df_acc['y_test']>=200000) & (df_acc['y_test']<=250000)]
                    acc_2[x] = len(df_20[df_20['error']<0.02])/len(df_20)
                    acc_5[x] = len(df_20[df_20['error']<0.05])/len(df_20)
                    acc_10[x] = len(df_20[df_20['error']<0.1])/len(df_20)
                    acc_20[x] = len(df_20[df_20['error']<0.2])/len(df_20)
                    x += 1
                except: 
                    ctt.pop('df_20')
                    troubleshoot(ctt, df_acc)

            if 'df_25' in ctt:
                try:
                    df_25 = df_acc[(df_acc['y_test']>=250000) & (df_acc['y_test']<=300000)]
                    acc_2[x] = len(df_25[df_25['error']<0.02])/len(df_25)
                    acc_5[x] = len(df_25[df_25['error']<0.05])/len(df_25)
                    acc_10[x] = len(df_25[df_25['error']<0.1])/len(df_25)
                    acc_20[x] = len(df_25[df_25['error']<0.2])/len(df_25)
                    x += 1
                except: 
                    ctt.pop('df_25')
                    troubleshoot(ctt, df_acc)

            if 'df_30' in ctt:
                try:
                    df_30 = df_acc[(df_acc['y_test']>=300000) & (df_acc['y_test']<=350000)]
                    acc_2[x] = len(df_30[df_30['error']<0.02])/len(df_30)
                    acc_5[x] = len(df_30[df_30['error']<0.05])/len(df_30)
                    acc_10[x] = len(df_30[df_30['error']<0.1])/len(df_30)
                    acc_20[x] = len(df_30[df_30['error']<0.2])/len(df_30)
                    x += 1
                except: 
                    ctt.pop('df_30')
                    troubleshoot(ctt, df_acc)

            if 'df_35' in ctt:
                try:
                    df_35 = df_acc[(df_acc['y_test']>=350000) & (df_acc['y_test']<=400000)]
                    acc_2[x] = len(df_35[df_35['error']<0.02])/len(df_35)
                    acc_5[x] = len(df_35[df_35['error']<0.05])/len(df_35)
                    acc_10[x] = len(df_35[df_35['error']<0.1])/len(df_35)
                    acc_20[x] = len(df_35[df_35['error']<0.2])/len(df_35)
                    x += 1
                except: 
                    ctt.pop('df_35')
                    troubleshoot(ctt, df_acc)

            if 'df_40' in ctt:
                try:
                    df_40 = df_acc[df_acc['y_test']>=400000]
                    acc_2[x] = len(df_40[df_40['error']<0.02])/len(df_40)
                    acc_5[x] = len(df_40[df_40['error']<0.05])/len(df_40)
                    acc_10[x] = len(df_40[df_40['error']<0.1])/len(df_40)
                    acc_20[x] = len(df_40[df_40['error']<0.2])/len(df_40)
                    x += 1
                except: 
                    ctt.pop('df_40')
                    troubleshoot(ctt, df_acc)

            return acc_2, acc_5, acc_10, acc_20, ctt

        acc_2, acc_5, acc_10, acc_20, ctt = troubleshoot(ctt, df_acc)

        acc = pd.DataFrame(
            {'acc_2': acc_2,
            'acc_5': acc_5,
            'acc_10': acc_10,
             'acc_20': acc_20,
            })

        a = []
        if 'df_0' in ctt:
            a.append('<50k')
        if 'df_5' in ctt:
            a.append('50k-100k')
        if 'df_10' in ctt:
            a.append('100k-150k')
        if 'df_15' in ctt:
            a.append('150k-200k')
        if 'df_20' in ctt:
            a.append('200k-250k')
        if 'df_25' in ctt:
            a.append('250k-300k')
        if 'df_30' in ctt:
            a.append('300k-350k')
        if 'df_35' in ctt:
            a.append('350k-400k')
        if 'df_40' in ctt:
            a.append('>400k')

        acc['value bands'] = a

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (20,10))
        ax.scatter(acc['acc_2'], acc['value bands'],label='err +-2%', 
                   s=80
                  )
        ax.scatter(acc['acc_5'], acc['value bands'],label='err +-5%', 
                   s=80
                  )
        ax.scatter(acc['acc_10'], acc['value bands'],label='err +-10%', 
                   s=80
                  )
        ax.scatter(acc['acc_20'], acc['value bands'],label='err +-20%', 
                   s=80
                  )

        for i in range(0, len(ctt)):
            plt.plot([acc['acc_2'][i], acc['acc_10'][i]], [[i]*5,[i]*5], 'grey')
            plt.plot([acc['acc_5'][i], acc['acc_20'][i]], [[i]*5,[i]*5], 'grey')

        plt.legend(fontsize=15)
        ax.set_xlim(0, 1)
        ax.set_xticklabels([0, .20, .40, .60, .80, 1], rotation=0, fontsize=15)
        ax.set_yticklabels(acc['value bands'], rotation=0, fontsize=15)
        ax.set_xlabel('Percent of valuations within +-5%, +-20%')
        ax.set_ylabel('Price range of properties $USD')
        plt.title('Accuracy by Value ' + model_name,fontsize=20)
        plt.show()

        plt.savefig('/dbfs/FileStore/artifacts/AccuracyByPriceRange.png')

        return df_acc


