from evalml.pipelines.components.transformers import Transformer
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types




class LookupTableTransformer(Transformer):
    """Transformer to Lookup Preprocessed Aggregations"""
    name = "Lookup Table Transformer"
    hyperparameter_ranges = {}

    def __init__(self, 
                 random_state=0, 
                 lookup_table=None,
                 **kwargs):
      
        self.lookup_table = lookup_table
        parameters = {}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)
        
    def fit(self, X, y=None):
        "dummy fit function to meet Transformer class reqs"
        return self

    def transform(self, X, y=None):
        """Transforms input data by Lookup of Preprocessed Aggregations by matching postcode
        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Targets
        Returns:
            pd.DataFrame: Transformed X
        """
        if not isinstance(X, pd.DataFrame): 
          X = pd.DataFrame(X)
          
        agg_data = self.lookup_table[self.lookup_table['POSTCODE'].isin(list(set(X['POSTCODE'].values)))]
        
        property_features = ['FLOOR_LEVEL_e','NUMBER_HEATED_ROOMS_e','TOTAL_FLOOR_AREA_e']
        agg_data_cols = [col for col in agg_data.columns if col not in property_features]
        agg_data = agg_data[agg_data_cols]
        agg_data_m = pd.merge(X, agg_data, left='POSTCODE_e', right='POSTCODE')
        
        agg_features = []
        X_t = pd.DataFrame({})
        for col in property_features:
          agg_cols = [c for c in agg_data.columns if col in c and c != col]
          for agg_col in agg_cols:
            X_t['diff_'+agg_col] = agg_data_m[col] - agg_data_m[agg_col]

        X_t['Latitude_m'] = agg_data_m['Latitude_m']
        X_t['Longitude_m'] = agg_data_m['Longitude_m']
        X_t['PROPERTY_TYPE_e'] = agg_data_m['PROPERTY_TYPE_e']
        X_t['POSTTOWN_e'] = agg_data_m['POSTTOWN_e']

        return X_t



