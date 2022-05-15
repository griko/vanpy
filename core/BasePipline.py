import logging
from abc import ABC
from dataclasses import dataclass
from logging import Logger
from typing import Dict, List, Tuple
from yaml import YAMLObject
from core.PiplineComponent import PipelineComponent
import pandas as pd


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

    def process(self, input_dir: str = '', df: pd.DataFrame = None) -> Tuple[str, pd.DataFrame]:
        process_dir = input_dir
        process_df = df
        for component in self.components:
            self.logger.info(f'Processing with {component.get_name()}')
            process_dir, process_df = component.process(process_dir, process_df)
        return process_dir, process_df
