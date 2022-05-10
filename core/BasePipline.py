import logging
from abc import ABC
from dataclasses import dataclass
from logging import Logger
from typing import Dict, List

from yaml import YAMLObject

from core.PiplineComponent import PipelineComponent


@dataclass
class BasePipeline(ABC):
    components_mapper: Dict
    components: List[PipelineComponent]
    logger: Logger

    def __init__(self, components: List[str], config: YAMLObject):
        self.components = []
        for component in components:
            c = self.components_mapper[component](config)
            self.components.append(c)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_components(self) -> List[PipelineComponent]:
        return self.components

    def process(self):
        pass