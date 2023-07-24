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
    """
    The main class for processing data through a sequence of pipelines.

    :param pipelines: List of pipelines (subclasses of `BasePipeline`) to process data through
    :type pipelines: List[BasePipeline]
    :param logger: Logger instance to record events during pipeline processing
    :type logger: Logger
    :param preprocessed_files_dir: Directory where preprocessed files are stored
    :type preprocessed_files_dir: str
    :param speaker_classification_df: Dataframe containing speaker classification results
    :type speaker_classification_df: pd.DataFrame
    :param segment_classification_df: Dataframe containing segment classification results
    :type segment_classification_df: pd.DataFrame
    """
    pipelines: List[BasePipeline]
    logger: Logger
    preprocessed_files_dir: str
    speaker_classification_df: pd.DataFrame
    segment_classification_df: pd.DataFrame

    def __init__(self, components: List[PipelineComponent] = None, pipelines: List[BasePipeline] = None,
                 config: YAMLObject = None):
        """
        Initializes the Pipeline object with a list of pipeline components, pipelines, or YAML configuration.

        :param components: List of pipeline components (instances of `PipelineComponent`) to add to the pipeline
        :type components: List[PipelineComponent], optional
        :param pipelines: List of pipelines (instances of `BasePipeline`) to add to the pipeline
        :type pipelines: List[BasePipeline], optional
        :param config: YAML configuration for the pipeline
        :type config: YAMLObject, optional
        """
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
        """
        Processes the input data through all pipelines in the sequence.

        :param initial_payload: Initial payload to be processed
        :type initial_payload: ComponentPayload, optional
        :return: Processed payload after all pipelines
        :rtype: ComponentPayload
        """
        process_dir = self.input_dir
        cp: ComponentPayload = ComponentPayload(input_path=process_dir)
        if initial_payload is not None:
            cp = initial_payload

        for pipeline in self.pipelines:
            if pipeline is not None:
                cp = pipeline.process(cp)

        return cp

    @staticmethod
    def generate_pipelines_from_components(components: List[PipelineComponent], config: YAMLObject = None):
        """
        Static method to generate a sequence of pipelines from a list of pipeline components.

        :param components: List of pipeline components (instances of `PipelineComponent`) to add to the pipelines
        :type components: List[PipelineComponent]
        :param config: YAML configuration for the pipeline
        :type config: YAMLObject, optional
        :return: List of pipelines generated from the components
        :rtype: List[BasePipeline]
        """
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
        """
        Static method to generate a pipeline from a list of pipeline components.

        :param components: List of pipeline components (instances of `PipelineComponent`) to add to the pipeline
        :type components: List[PipelineComponent]
        :param pipeline_class: The class of pipeline to generate
        :type pipeline_class: BasePipeline
        :param config: YAML configuration for the pipeline
        :type config: YAMLObject, optional
        :return: A pipeline generated from the components
        :rtype: BasePipeline
        """
        pipeline_components = []
        for component in components:
            if component in pipeline_class.components_mapper:
                pipeline_components.append(component)
        pipeline = None
        if pipeline_components:
            pipeline = pipeline_class(pipeline_components, config=config)
        return pipeline
