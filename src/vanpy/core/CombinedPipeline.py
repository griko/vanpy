from dataclasses import dataclass
from logging import Logger
import pandas as pd
import logging
from typing import List

from vanpy.core.BasePipline import BasePipeline
from vanpy.core.ComponentPayload import ComponentPayload
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

    def process(self, initial_payload: ComponentPayload = None) -> ComponentPayload:
        process_dir = self.input_dir
        cp: ComponentPayload = ComponentPayload(input_path=process_dir)
        if initial_payload:
            cp = initial_payload

        for pipeline in self.pipelines:
            if pipeline is not None:
                cp = pipeline.process(cp)

        return cp
