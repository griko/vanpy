from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd


@dataclass
class ComponentPayload:
    metadata: dict
    df: pd.DataFrame

    def __init__(self, input_path: str = '', metadata: Dict = None, df: pd.DataFrame = None):
        if input_path:
            self.metadata = {'input_path': input_path, 'paths_column': '', 'all_paths_columns': [],
                             'meta_columns': [], 'feature_columns': [], 'classification_columns': []}
            self.df = pd.DataFrame()
        if metadata:
            self.metadata = metadata
        if pd.DataFrame:
            self.df = df

    def unpack(self) -> Tuple[Dict, pd.DataFrame]:
        return self.metadata, self.df

    def get_columns(self, all_paths_columns=False, meta_columns=False):
        if not all_paths_columns:
            columns = [self.metadata['paths_column']]
        else:
            columns = self.metadata['all_paths_columns']
        if meta_columns:
            columns.extend(self.metadata['meta_columns'])
        return columns

    def get_declared_columns(self, ext_columns: List[str], all_paths_columns=False, meta_columns=False):
        columns = self.get_columns(all_paths_columns, meta_columns)
        for cols in ext_columns:
            columns.extend(self.metadata[cols])
        columns = list(set(columns) & set(self.df.columns))
        return self.df[columns]

    def get_features_df(self, all_paths_columns=False, meta_columns=False):
        return self.get_declared_columns(['feature_columns'], all_paths_columns, meta_columns)

    def get_classification_df(self, all_paths_columns=False, meta_columns=False):
        return self.get_declared_columns(['classification_columns'], all_paths_columns, meta_columns)

    def get_full_df(self, all_paths_columns=False, meta_columns=False):
        return self.get_declared_columns(['feature_columns', 'classification_columns'], all_paths_columns, meta_columns)

    def remove_redundant_index_columns(self):
        for c in self.df.columns:
            if c.startswith('Unnamed') or c == '':
                self.df.drop([c], axis=1, inplace=True)

