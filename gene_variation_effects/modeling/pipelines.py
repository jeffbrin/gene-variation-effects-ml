
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np



class NNPipeLine():
    def __init__(self, column_names : list[str]) -> None:
        # Design matrix index to feature str mapping
        self.var_to_idx = dict(zip(column_names, range(len(column_names))))

        onehot_features = ['Type', 'ChromosomeAccession', 'Chromosome']
        self.onehot_idx = [self.var_to_idx.get(key) for key in onehot_features]
        
        emb_features = ['GeneSymbol', 'Cytogenetic', 'ReferenceAlleleVCF', 'AlternateAlleleVCF']
        self.emb_idx = [self.var_to_idx.get(key) for key in emb_features]

        numerical_features = 'VariantLength'
        self.numerical_idx = [self.var_to_idx.get(numerical_features)]

    def fit_for_all(self, X_train : np.ndarray) -> tuple[np.ndarray, Pipeline]:
        """
        Fit transformations on the X training set for all features

        Args:
            X_train (np.ndarray): Design matrix

        Returns:
            tuple[np.ndarray, Pipeline]: Transformed X_train and fit pipeline
        """
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy = "constant", fill_value = "<MISSING>"))
        ])
        return pipe.fit_transform(X_train), pipe

    def fit_feature_transformations(self, X_train : np.ndarray) -> tuple[np.ndarray, Pipeline, Pipeline]:
        """
        Fit transformations on the X_training set for specific features

        Args:
            X_train (np.ndarray): Design matrix

        Returns:
            tuple[np.ndarray, Pipeline, Pipeline]: Transformed X_train and fit pipeline
        """        
        onehot_pipe = Pipeline([('onehot', OneHotEncoder(sparse = False))])
        emb_pipe = Pipeline([('label_encode', OrdinalEncoder(handle_unknown = 'error'))])
        feature_scaling = Pipeline([('norm', MinMaxScaler())])
        
        feature_processor = ColumnTransformer(
            transformers = [
                ('low_cardinality', onehot_pipe, self.onehot_idx),
                ('high_cardinality', emb_pipe, self.emb_idx),
                ('numerical', feature_scaling, self.numerical_idx)
            ],
            remainder = 'passthrough'
        )
        return feature_processor.fit_transform(X_train), feature_processor
    
    