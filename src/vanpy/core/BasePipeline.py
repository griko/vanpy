import logging
from abc import ABC
from dataclasses import dataclass
from logging import Logger
from typing import Dict, List
from yaml import YAMLObject

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent


@dataclass
class BasePipeline(ABC):
    """
    BasePipeline

    This is an abstract base class for a pipeline. It holds a list of PipelineComponent objects and processes the input
    payload through each component in order. The components_mapper attribute should be set in a subclass to map
    component names to their corresponding class. The process method is used to process the input payload and pass it
    through each component.
    """
    components_mapper: Dict  # A dictionary that maps component names to their corresponding class
    components: List[PipelineComponent]  # A list of `PipelineComponent` objects that the input payload will be passed through
    logger: Logger  # A logger for the pipeline to log events and progress

    def __init__(self, components: List[str], config: YAMLObject):
        """
        Initialize the pipeline with a list of components and a configuration object.

        :param components: A list of component names
        :param config: A YAML object containing configuration for the pipeline and its components
        """
        self.components = []
        for component in components:
            c = self.components_mapper[component](config)
            self.components.append(c)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_components(self) -> List[PipelineComponent]:
        """
        Returns the list of `PipelineComponent` objects in the pipeline.

        :return: A list of `PipelineComponent` objects
        """
        return self.components

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process the input payload by passing it through each component in the pipeline.

        :param input_payload: A `ComponentPayload` object to be processed
        :return: A processed `ComponentPayload` object
        """
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
