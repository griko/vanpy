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
    Abstract base class for all pipeline types in the system.

    Provides core functionality for managing and executing sequences of processing
    components. Each pipeline maintains a list of components and processes data
    through them in order.

    :ivar components_mapper: Maps component names to their implementing classes.
    :ivar components: List of instantiated pipeline component objects.
    :ivar logger: Logger instance for recording pipeline events.
    """
    components_mapper: Dict  # A dictionary that maps component names to their corresponding class
    components: List[PipelineComponent]  # A list of `PipelineComponent` objects that the input payload will be passed through
    logger: Logger  # A logger for the pipeline to log events and progress

    def __init__(self, components: List[str], config: YAMLObject):
        """
        Initialize pipeline with components and configuration.

        :param components: List of component names to include in pipeline.
        :param config: Configuration parameters for pipeline and components.
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
        Process input data through all pipeline components sequentially.

        :param input_payload: Data to be processed through the pipeline.
        :return: Processed data after passing through all components.
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
