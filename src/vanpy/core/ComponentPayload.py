from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
import pickle


@dataclass
class ComponentPayload:
    """
    Container for data and metadata passed between pipeline components.

    Manages a DataFrame containing the actual data along with metadata describing
    columns, paths, features, and classifications.

    :ivar metadata: Dictionary containing metadata about the payload contents.
    :ivar df: DataFrame containing the actual data being processed.
    """
    metadata: dict
    df: pd.DataFrame

    def __init__(self, input_path: str = '', metadata: Dict = None, df: pd.DataFrame = None):
        """
        Initialize a new payload container.

        :param input_path: Path to input data directory.
        :param metadata: Initial metadata dictionary.
        :param df: Initial DataFrame.
        :raises AttributeError: If neither input_path nor paths_column is provided.
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
                "You must supply at least input_path or not empty metadata['paths_column'] when initializing ComponentPayload")
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
        :param meta_columns: whether to include meta columns in the returned list
        :return: list of column names
        """
        if not all_paths_columns:
            columns = [self.metadata['paths_column']]
        else:
            columns = self.metadata['all_paths_columns']
        if meta_columns:
            columns.extend(self.metadata['meta_columns'])
        return columns

    def get_declared_columns(self, ext_columns: List[str], all_paths_columns=False, meta_columns=False) -> pd.DataFrame:
        """
        Returns a payload's dataframe containing the specified columns.

        :param ext_columns: the list of columns to include in the returned dataframe
        :param all_paths_columns: whether to include all paths columns in the returned dataframe 
        :param meta_columns: whether to include meta columns in the returned dataframe
        :return: a dataframe containing the specified columns
        """
        columns = self.get_columns(all_paths_columns, meta_columns)
        for cols in ext_columns:
            columns.extend(self.metadata[cols])
        columns = [c for c in columns if c in self.df.columns]
        return self.df[columns]

    def get_features_df(self, all_paths_columns=False, meta_columns=False):
        """
        Extract feature columns from the payload.

        :param all_paths_columns: Whether to include all path columns.
        :param meta_columns: Whether to include metadata columns.
        :return: DataFrame containing only feature columns.
        """
        return self.get_declared_columns(['feature_columns'], all_paths_columns, meta_columns)

    def get_classification_df(self, all_paths_columns=False, meta_columns=False):
        """
        Extract classification columns from the payload.

        :param all_paths_columns: Whether to include all path columns.
        :param meta_columns: Whether to include metadata columns.
        :return: DataFrame containing only classification columns.
        """
        return self.get_declared_columns(['classification_columns'], all_paths_columns, meta_columns)

    def get_full_df(self, all_paths_columns=False, meta_columns=False):
        """
        Returns a dataframe containing the feature and classification columns of the payload.

        :param all_paths_columns: whether to include all paths columns in the returned dataframe
        :param meta_columns: whether to include meta columns in the returned dataframe
        :return: a dataframe containing the feature and classification columns
        """
        return self.get_declared_columns(['feature_columns', 'classification_columns'], all_paths_columns, meta_columns)

    def remove_redundant_index_columns(self):
        """
        Removes any columns from the payload's dataframe that have a name that starts with "Unnamed" or is an empty string.
        """
        for c in self.df.columns:
            if c.startswith('Unnamed') or c == '':
                self.df.drop([c], axis=1, inplace=True)

    def save(self, output_dir: str, name: str = 'payload', index=False):
        """
        Saves the payload's dataframe to a csv file and metadata as pickle in the given output directory.

        :param output_dir: the output directory
        :param name: the name of the output file
        :param index: whether to include the index in the output file
        """
        self.df.to_csv(f'{output_dir}/{name}.csv', index=index)
        pickle.dump(self.metadata, open(f'{output_dir}/{name}.pkl', 'wb'))


