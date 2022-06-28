from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
from yaml import YAMLObject
from logging import Logger
import logging

from audio_pipeline.core.ComponentPayload import ComponentPayload


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
