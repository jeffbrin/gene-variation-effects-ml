
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np



class NNPipeLine():
    def __init__(self, column_names : list[str], onehot_features: list[str], emb_features: list[str], numerical_features: list[str]) -> None:
        """
        Constructs the pipeline with the given features.

        Parameters
        ----------
        column_names : list[str]
            All column names in the order they appear in the dataset.
        onehot_features : list[str]
            Column names for features which should use onehot encoding.
        emb_features : list[str]
            Column names for features which should use embedded encoding.
        numerical_features : list[str]
            Column names for numerical features.
        """
        # Design matrix index to feature str mapping
        self.var_to_idx = dict(zip(column_names, range(len(column_names))))

        self.onehot_idx = [self.var_to_idx.get(key) for key in onehot_features]
        self.emb_idx = [self.var_to_idx.get(key) for key in emb_features]
        self.numerical_idx = [self.var_to_idx.get(key) for key in numerical_features]

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

    def fit_feature_transformations(self, X_train : np.ndarray) -> tuple[np.ndarray, ColumnTransformer]:
        """
        Fit transformations on the X_training set for specific features

        Args:
            X_train (np.ndarray): Design matrix

        Returns:
            tuple[np.ndarray, Pipeline, Pipeline]: Transformed X_train and fit pipeline
        """        
        onehot_pipe = Pipeline([('onehot', OneHotEncoder(sparse_output = False))])
        emb_pipe = Pipeline([('label_encode', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=-1))])
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