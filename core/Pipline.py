from typing import Dict, List, Callable, Tuple
from core.PipelineClassificationComponent import PipelineClassificationComponent
from core.PipelinePreprocessingComponent import PipelinePreprocessingComponent
from core.preprocess_components.voice_music_separator.INAVoiceSeparator import INAVoiceSeparator
from core.preprocess_components.encoding_converter.WAVConverter import WAVConverter
from yaml import YAMLObject
import functools
from dataclasses import dataclass
import logging
from logging import Logger
import pandas as pd
from core.PiplineComponent import PipelineComponent


class PreprocessPipeline:
    components_mapper = {
        'wav_converter': WAVConverter,
        'ina_speech_segmenter': INAVoiceSeparator
    }
    components: List[PipelineComponent]

    def __init__(self, components: List[str], config: YAMLObject):
        self.components = []
        for component in components:
            c = PreprocessPipeline.components_mapper[component](config)
            self.components.append(c)

    def get_components(self) -> List[PipelineComponent]:
        return self.components

    def process(self) -> str:
        last_output_dir = ''
        for component in self.components:
            last_output_dir = component.process()
        return last_output_dir


class SpeakerClassificationPipeline():

    # components_mapper = {
    #     'common_voices_age':
    # }
    def __init__(self, components:List[str]):
        self.components = []
        for component in components:
            self.components.append(SpeakerClassificationPipeline.components_mapper[component]())
    pass


class SegmentClassificationPipeline(PipelineClassificationComponent):
    pass


@dataclass
class Pipeline:
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
                preprocessed_files_dir = component.process()
        # TODO: run speaker classification pipeline

        # TODO: run speaker classification pipeline

        return preprocessed_files_dir, None, None
