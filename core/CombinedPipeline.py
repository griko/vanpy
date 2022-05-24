from dataclasses import dataclass
from logging import Logger
import pandas as pd
import logging
from typing import Tuple, List

from core.BasePipline import BasePipeline
from core.PiplineComponent import ComponentPayload
from yaml import YAMLObject


@dataclass
class CombinedPipeline:
    pipelines: List[BasePipeline]
    logger: Logger
    preprocessed_files_dir: str
    speaker_classification_df: pd.DataFrame
    segment_classification_df: pd.DataFrame

    def __init__(self, pipelines: List[BasePipeline] = None, config: YAMLObject = None):
        self.config = config
        self.input_dir = self.config['input_dir']
        self.pipelines = pipelines
        self.logger = logging.getLogger('Combined Pipeline')

    def process(self) -> ComponentPayload:
        process_dir = self.input_dir
        cp: ComponentPayload = ComponentPayload(input_path=process_dir)

        for pipeline in self.pipelines:
            cp = pipeline.process(cp)

        return cp
