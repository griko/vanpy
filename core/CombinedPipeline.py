from dataclasses import dataclass
from logging import Logger
import pandas as pd
import logging
from typing import Tuple
from core.Pipline import SegmentClassificationPipeline
from core.SpeakerClassificationPipline import SpeakerClassificationPipeline
from core.PreprocessPipline import PreprocessPipeline
from yaml import YAMLObject


@dataclass
class CombinedPipeline:
    pre_process_pipline: PreprocessPipeline
    speaker_clf_pipeline: SpeakerClassificationPipeline
    segment_clf_pipeline: SegmentClassificationPipeline
    logger: Logger
    preprocessed_files_dir: str
    speaker_classification_df: pd.DataFrame
    segment_classification_df: pd.DataFrame

    def __init__(self, pre_process_pipline: PreprocessPipeline = None,
                 speaker_clf_pipeline: SpeakerClassificationPipeline = None,
                 segment_clf_pipeline: SegmentClassificationPipeline = None, config: YAMLObject = None):
        self.config = config
        self.pre_process_pipline = pre_process_pipline
        self.speaker_clf_pipeline = speaker_clf_pipeline
        self.segment_clf_pipeline = segment_clf_pipeline
        self.logger = logging.getLogger('Pipeline')

    def process(self) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        preprocessed_files_dir = ''
        # run pre-process pipeline
        if self.pre_process_pipline:
            for component in self.pre_process_pipline.get_components():
                preprocessed_files_dir, preprocessed_df = component.process(preprocessed_files_dir, preprocessed_df)
        # TODO: run speaker classification pipeline
        if self.speaker_clf_pipeline:
            speaker_clf_df = pd.DataFrame()
            for component in self.speaker_clf_pipeline.get_components():
                speaker_clf_df = component.process(speaker_clf_df)
        # TODO: run speaker classification pipeline

        return preprocessed_files_dir, speaker_clf_df, None
