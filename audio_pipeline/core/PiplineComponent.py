from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
from yaml import YAMLObject
from logging import Logger
import logging
import pandas as pd


@dataclass
class ComponentPayload:
    features: Dict
    df: pd.DataFrame

    def __init__(self, input_path: str = '', features: Dict = None, df: pd.DataFrame = None):
        if input_path:
            self.features = {}
            self.features['input_path'] = input_path
            self.features['paths_column'] = ''
            self.features['feature_columns'] = []
            self.features['classification_columns'] = []
            self.df = pd.DataFrame()
        if features:
            self.features = features
        if pd.DataFrame:
            self.df = df

    def unpack(self) -> Tuple[Dict, pd.DataFrame]:
        return self.features, self.df

    def get_features_df(self):
        columns = [self.features['paths_column']]
        columns.extend(self.features['feature_columns'])
        return self.df[columns]

    def get_classification_df(self):
        columns = [self.features['paths_column']]
        columns.extend(self.features['classification_columns'])
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
    def process(self, input_object: ComponentPayload) -> ComponentPayload:
        pass
