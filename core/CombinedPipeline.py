from dataclasses import dataclass
from logging import Logger
import pandas as pd
import logging
from typing import Tuple
from core.Pipline import SegmentClassificationPipeline
from core.ClassificationPipline import ClassificationPipeline
from core.PiplineComponent import ComponentPayload
from core.PreprocessPipline import PreprocessPipeline
from yaml import YAMLObject


@dataclass
class CombinedPipeline:
    pre_process_pipline: PreprocessPipeline
    speaker_clf_pipeline: ClassificationPipeline
    segment_clf_pipeline: SegmentClassificationPipeline
    logger: Logger
    preprocessed_files_dir: str
    speaker_classification_df: pd.DataFrame
    segment_classification_df: pd.DataFrame

    def __init__(self, pre_process_pipline: PreprocessPipeline = None,
                 speaker_clf_pipeline: ClassificationPipeline = None,
                 segment_clf_pipeline: SegmentClassificationPipeline = None, config: YAMLObject = None):
        self.config = config
        self.input_dir = self.config['input_dir']
        self.pre_process_pipline = pre_process_pipline
        self.speaker_clf_pipeline = speaker_clf_pipeline
        self.segment_clf_pipeline = segment_clf_pipeline
        self.logger = logging.getLogger('Pipeline')

    def process(self) -> ComponentPayload:
        process_dir = self.input_dir
        process_df = pd.DataFrame()
        cp: ComponentPayload = ComponentPayload(features={'input_path': process_dir}, df=process_df)
        # run pre-process pipeline
        if self.pre_process_pipline:
            cp = self.pre_process_pipline.process(cp)
            # for component in self.pre_process_pipline.get_components():
            #     process_dir, process_df = component.process(process_dir, process_df)
        # TODO: run speaker classification pipeline
        if self.speaker_clf_pipeline:
            cp = self.speaker_clf_pipeline.process(cp)
            # for component in self.speaker_clf_pipeline.get_components():
            #     speaker_clf_df = component.process(process_dir, process_df)
        # TODO: run speaker classification pipeline

        return cp
