from dataclasses import dataclass
from logging import Logger
import pandas as pd
import logging
from typing import List

from vanpy.core.BasePipeline import BasePipeline
from vanpy.core.ComponentPayload import ComponentPayload
from yaml import YAMLObject

from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.core.PreprocessPipeline import PreprocessPipeline
from vanpy.core.FeatureExtractionPipeline import FeatureExtractionPipeline
from vanpy.core.ClassificationPipeline import ClassificationPipeline


@dataclass
class Pipeline:
    pipelines: List[BasePipeline]
    logger: Logger
    preprocessed_files_dir: str
    speaker_classification_df: pd.DataFrame
    segment_classification_df: pd.DataFrame

    def __init__(self, components: List[PipelineComponent] = None, pipelines: List[BasePipeline] = None,
                 config: YAMLObject = None):
        self.config = config
        self.input_dir = self.config['input_dir']
        if pipelines:
            self.pipelines = pipelines
        elif components:
            self.pipelines = self.generate_pipelines_from_components(components, self.config)
        else:
            raise AttributeError("You have supplied both empty components list and pipelines list")
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

    @staticmethod
    def generate_pipelines_from_components(components: List[PipelineComponent], config: YAMLObject = None):
        preprocessing_pipeline = Pipeline.generate_pipeline_from_components(components=components,
                                                                            pipeline_class=PreprocessPipeline,
                                                                            config=config)
        feature_extraction_pipeline = Pipeline.generate_pipeline_from_components(components=components,
                                                                                 pipeline_class=FeatureExtractionPipeline,
                                                                                 config=config)
        classification_and_stt_pipeline = Pipeline.generate_pipeline_from_components(components=components,
                                                                                     pipeline_class=ClassificationPipeline,
                                                                                     config=config)
        pipelines = []
        if preprocessing_pipeline:
            pipelines.append(preprocessing_pipeline)
        if feature_extraction_pipeline:
            pipelines.append(feature_extraction_pipeline)
        if classification_and_stt_pipeline:
            pipelines.append(classification_and_stt_pipeline)
        return pipelines

    @staticmethod
    def generate_pipeline_from_components(components: List[PipelineComponent], pipeline_class: BasePipeline,
                                          config: YAMLObject = None):
        pipeline_components = []
        for component in components:
            if component in pipeline_class.components_mapper:
                pipeline_components.append(component)
        pipeline = None
        if pipeline_components:
            pipeline = pipeline_class(pipeline_components, config=config)
        return pipeline
