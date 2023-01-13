import logging
from abc import ABC
from dataclasses import dataclass
from logging import Logger
from typing import Dict, List
from yaml import YAMLObject

from src.vanpy.core.ComponentPayload import ComponentPayload
from src.vanpy.core.PipelineComponent import PipelineComponent


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

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        payload_object = input_payload
        for component in self.components:
            self.logger.info(f'Processing with {component.get_name()}')
            # if inspect.iscoroutinefunction(component.process):
            #     payload_object = await component.process(payload_object)
            # else:
            payload_object = component.process(payload_object)
            # payload_object.remove_redundant_index_columns()  # get rid of "Unnamed XX" columns
            component.save_component_payload(payload_object)  # save intermediate results, if enabled

        return payload_object
