
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from pandas import Series, DataFrame
import torch
import pandas as pd


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
        return feature_processor.fit_transform(X_train).astype(float), feature_processor    
    
    def split_multi_value_feature(self, data: DataFrame, feature: str) -> DataFrame:
        # Split each GeneSymbol into multiple rows
        column_index = list(data.columns).index(feature)
        s = data[feature].str.split(';')
        s = s.explode()
        s.name = feature
        data.drop(columns=[feature], inplace=True)
        data = data.join(s)
        
        feature_column = data[feature]
        data.drop(columns=[feature], inplace=True)
        data.insert(column_index, feature, feature_column)
        return data
    
    # Merge rows which were split earlier on gene symbol
    def combine_duplicated_genesymbol_rows(self, data: np.ndarray, genesymbol_column_index: int, group_column: int) -> tuple[torch.Tensor, list[str]]:
        columns = list(range(data.shape[1]))
        columns[genesymbol_column_index] = 'GeneSymbol'
        columns[-1] = group_column
        grouped_df = pd.DataFrame(data, columns=columns).groupby(by=[group_column])
        gene_symbol_lists = grouped_df['GeneSymbol'].apply(lambda g: ";".join(str(x) for x in g))
        output_df = grouped_df.mean()
        encoded_gene_lists, unique_gene_lists = pd.factorize(gene_symbol_lists)

        output_df['GeneSymbol'] = encoded_gene_lists
        columns.remove(group_column)
        return torch.Tensor(output_df[columns].to_numpy()), unique_gene_lists
    
    def split_multi_features_from_data_output(self, data: np.ndarray, data_columns: list[str], group_column):
        data_df = pd.DataFrame(data, columns=data_columns)
        data_df = self.split_multi_value_feature(data_df, 'GeneSymbol')
        data_df[group_column] = data_df.index.astype(int)

        return data_df.to_numpy(), data_df.columns
    
    def get_embedding_input_sizes(feature_processor: ColumnTransformer) -> list[int]:
        # Getting input sizes for embedding layer
        embedding_processor = feature_processor.named_transformers_['high_cardinality']
        label_encoder = embedding_processor.named_steps['label_encode']

        extra_cat_for_potential_unknown = 1 if label_encoder.handle_unknown == 'use_encoded_value' else 0
        if hasattr(label_encoder, "categories_"):
            return [cat.size + extra_cat_for_potential_unknown for cat in label_encoder.categories_]
        else:
            return []

        