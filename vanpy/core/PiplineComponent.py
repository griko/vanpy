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

    def latent_info_log(self, message: str, iteration: int, last_item: bool = False) -> None:
        log_each_x_records = 1
        if self.config['log_each_x_records']:
            log_each_x_records = self.config['log_each_x_records']
        last_item = False
        if self.config['items_in_paths_list']:
            last_item = iteration == self.config['items_in_paths_list']
        if iteration % log_each_x_records == 0 or last_item:
            self.logger.info(message)

    def import_config(self, yaml_config: YAMLObject) -> Dict:
        config = yaml_config[self.component_type][self.component_name]
        config = {} if config is None else config
        for item in yaml_config:  # pass through all root level configs
            if isinstance(item, str) and item not in config:
                config[item] = yaml_config[item]
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
        self.get_logger().info(
            f'Called Saved payload {self.get_name(), "save_payload" in self.config and self.config["save_payload"]}, intermediate {intermediate}')
        if "save_payload" in self.config and self.config["save_payload"]:
            create_dirs_if_not_exist(self.config["intermediate_payload_path"])
            metadata, df = input_payload.unpack()
            with open(
                f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_metadata_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.pickle',
                'wb') as handle:
                pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # input_payload.get_classification_df(all_paths_columns=True, meta_columns=True).to_csv(f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_clf_df_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.csv')
            df.to_csv(
                f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_df_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.csv',
                index=False)
            self.get_logger().info(f'Saved payload in {self.config["intermediate_payload_path"]}')

    def save_intermediate_payload(self, i: int, input_payload: ComponentPayload):
        if 'save_payload_periodicity' in self.config and i % self.config['save_payload_periodicity'] == 0 and i > 0:
            self.save_component_payload(input_payload, intermediate=True)
