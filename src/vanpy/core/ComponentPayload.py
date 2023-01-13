from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd


@dataclass
class ComponentPayload:
    """
    A class that represents a container for payload (dataframe and metadata) passed between pipline components.
    """
    metadata: dict
    df: pd.DataFrame

    def __init__(self, input_path: str = '', metadata: Dict = None, df: pd.DataFrame = None):
        """
        Initializes the ComponentPayload class with the given input_path, metadata and dataframe.

        :param input_path: the input path of the data
        :type input_path: str
        :param metadata: the metadata of the data
        :type metadata: Dict
        :param df: the dataframe containing the data
        :type df: pd.DataFrame
        """
        self.metadata = metadata
        self.df = df
        if not self.metadata:
            self.metadata = {'input_path': '', 'paths_column': '', 'all_paths_columns': [],
                             'meta_columns': [], 'feature_columns': [], 'classification_columns': []}
        if input_path:
            self.metadata['input_path'] = input_path
        if ('input_path' not in self.metadata or self.metadata['input_path'] == '') and \
            ('paths_column' not in self.metadata or self.metadata['paths_column'] == ''):
            raise AttributeError(
                "You must supply at least input_path or metadata['paths_column'] when initializing ComponentPayload")
        for col in ['all_paths_columns', 'meta_columns', 'feature_columns', 'classification_columns']:
            if col not in self.metadata:
                self.metadata[col] = []
        if 'paths_column' in self.metadata and not self.metadata['all_paths_columns']:
            self.metadata['all_paths_columns'].append(self.metadata['paths_column'])
        if self.df is None:
            self.df = pd.DataFrame()

    def unpack(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Returns a tuple of payload's metadata and the dataframe.

        :return: tuple of metadata and the dataframe
        :rtype: Tuple[Dict, pd.DataFrame]
        """
        return self.metadata, self.df

    def get_columns(self, all_paths_columns=False, meta_columns=False):
        """
        Returns the list of column names stored in metadata, filtered based on the input parameters.

        :param all_paths_columns: whether to include all paths columns in the returned list
        :type all_paths_columns: bool
        :param meta_columns: whether to include meta columns in the returned list
        :type meta_columns: bool
        :return: list of column names
        :rtype: List[str]
        """
        if not all_paths_columns:
            columns = [self.metadata['paths_column']]
        else:
            columns = self.metadata['all_paths_columns']
        if meta_columns:
            columns.extend(self.metadata['meta_columns'])
        return columns

    def get_declared_columns(self, ext_columns: List[str], all_paths_columns=False, meta_columns=False):
        """
        Returns a payload's dataframe containing the specified columns.

        :param ext_columns: the list of columns to include in the returned dataframe
        :type ext_columns: List[str]
        :param all_paths_columns: whether to include all paths columns in the returned dataframe
        :type all_paths_columns: bool
        :param meta_columns: whether to include meta columns in the returned dataframe
        :type meta_columns: bool
        :return: a dataframe containing the specified columns
        :rtype: pd.DataFrame
        """
        columns = self.get_columns(all_paths_columns, meta_columns)
        for cols in ext_columns:
            columns.extend(self.metadata[cols])
        columns = list(set(columns) & set(self.df.columns))
        return self.df[columns]

    def get_features_df(self, all_paths_columns=False, meta_columns=False):
        """
        Returns a dataframe containing the feature columns of the payload.

        :param all_paths_columns: whether to include all paths columns in the returned dataframe
        :type all_paths_columns: bool
        :param meta_columns: whether to include meta columns in the returned dataframe
        :type meta_columns: bool
        :return: a dataframe containing the feature columns
        :rtype: pd.DataFrame
        """
        return self.get_declared_columns(['feature_columns'], all_paths_columns, meta_columns)

    def get_classification_df(self, all_paths_columns=False, meta_columns=False):
        """
        Returns a dataframe containing the classification columns of the payload.

        :param all_paths_columns: whether to include all paths columns in the returned dataframe
        :type all_paths_columns: bool
        :param meta_columns: whether to include meta columns in the returned dataframe
        :type meta_columns: bool
        :return: a dataframe containing the classification columns
        :rtype: pd.DataFrame
        """
        return self.get_declared_columns(['classification_columns'], all_paths_columns, meta_columns)

    def get_full_df(self, all_paths_columns=False, meta_columns=False):
        """
        Returns a dataframe containing the feature and classification columns of the payload.

        :param all_paths_columns: whether to include all paths columns in the returned dataframe
        :type all_paths_columns: bool
        :param meta_columns: whether to include meta columns in the returned dataframe
        :type meta_columns: bool
        :return: a dataframe containing the feature and classification columns
        :rtype: pd.DataFrame
        """
        return self.get_declared_columns(['feature_columns', 'classification_columns'], all_paths_columns, meta_columns)

    def remove_redundant_index_columns(self):
        """
        Removes any columns from the payload's dataframe that have a name that starts with "Unnamed" or is an empty string.
        """
        for c in self.df.columns:
            if c.startswith('Unnamed') or c == '':
                self.df.drop([c], axis=1, inplace=True)

