from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
from yaml import YAMLObject
from logging import Logger
import logging
import pickle
from datetime import datetime
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.utils.utils import create_dirs_if_not_exist


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
        config = yaml_config[self.component_type][self.component_name]
        config = {} if config is None else config
        config["intermediate_payload_path"] = yaml_config["intermediate_payload_path"]
        return config

    def get_logger(self) -> Logger:
        return logging.getLogger(f'{self.component_type} - {self.component_name}')

    def get_name(self) -> str:
        return self.component_name

    @abstractmethod
    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        pass

    # @staticmethod
    def save_component_payload(self, input_payload: ComponentPayload, intermediate=False) -> None:
        subscript = 'intermediate' if intermediate else 'final'
        if "save_payload" in self.config and self.config["save_payload"]:
            create_dirs_if_not_exist(self.config["intermediate_payload_path"])
            metadata, df = input_payload.unpack()
            with open(f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_metadata_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.pickle', 'wb') as handle:
                pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # input_payload.get_classification_df(all_paths_columns=True, meta_columns=True).to_csv(f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_clf_df_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.csv')
            input_payload.get_full_df(all_paths_columns=True, meta_columns=True).to_csv(
                f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_df_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.csv')

    def save_intermediate_payload(self, i: int, input_payload: ComponentPayload):
        if 'save_payload_periodicity' in self.config and i % self.config['save_payload_periodicity'] == 0:
            self.save_component_payload(input_payload, intermediate=True)
