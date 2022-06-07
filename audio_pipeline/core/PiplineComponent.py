from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
from yaml import YAMLObject
from logging import Logger
import logging
import pandas as pd


@dataclass
class ComponentPayload:
    metadata: Dict
    df: pd.DataFrame

    def __init__(self, input_path: str = '', metadata: Dict = None, df: pd.DataFrame = None):
        if input_path:
            self.metadata = {'input_path': input_path, 'paths_column': '', 'all_paths_columns': [],
                             'feature_columns': [], 'classification_columns': []}
            self.df = pd.DataFrame()
        if metadata:
            self.metadata = metadata
        if pd.DataFrame:
            self.df = df

    def unpack(self) -> Tuple[Dict, pd.DataFrame]:
        return self.metadata, self.df

    def get_features_df(self, all_paths_columns=False):
        if not all_paths_columns:
            columns = [self.metadata['paths_column']]
        else:
            columns = [self.metadata['all_paths_columns']]
        columns.extend(self.metadata['feature_columns'])
        return self.df[columns]

    def get_classification_df(self, all_paths_columns=False):
        if not all_paths_columns:
            columns = [self.metadata['paths_column']]
        else:
            columns = self.metadata['all_paths_columns']
        columns.extend(self.metadata['classification_columns'])
        return self.df[columns]


@dataclass
class PipelineComponent(ABC):
    component_type: str
    component_name: str
    config: Dict
    logger: Logger

    def __init__(self, component_type: str, component_name: str, yaml_config: YAMLObject):
        self.component_type = component_type
        self.component_name = component_name
        self.config = self.import_config(yaml_config)
        self.logger = self.get_logger()

    def import_config(self, yaml_config: YAMLObject) -> Dict:
        return yaml_config[self.component_type][self.component_name]

    def get_logger(self) -> Logger:
        return logging.getLogger(f'{self.component_type} - {self.component_name}')

    def get_name(self) -> str:
        return self.component_name

    @abstractmethod
    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        pass
