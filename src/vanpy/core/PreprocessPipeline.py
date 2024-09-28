from typing import List, Dict, Optional, Type
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline
from vanpy.core.PipelineComponent import PipelineComponent


class PreprocessPipeline(BasePipeline):
    """
    Class representing a preprocessing pipeline, which is a specific type of BasePipeline.
    It comprises various predefined components for audio preprocessing.

    :ivar components_mapper: Dictionary mapping component names to component classes or None.
        Each key is a string (the name of the component), and each value is either None or an instance of a class
        that inherits from PipelineComponent.
    :vartype components_mapper: Dict[str, Optional[PipelineComponent]]
    """
    components_mapper: Dict[str, Optional[PipelineComponent]] = {
        'file_mapper': None,
        'wav_converter': None,
        'wav_splitter': None,
        'ina_speech_segmenter': None,
        'pyannote_vad': None,
        'pyannote_sd': None,
        'silero_vad': None,
        'metricgan_se': None,
        'sepformer_se': None,
    }

    def __init__(self, components: List[str], config: YAMLObject):
        """
        Initializes the PreprocessPipeline object with the specified components and YAML configuration.

        The components list should be a list of strings where each string is a key in the `components_mapper`
        dictionary. The function then replaces the None value for each key in the `components_mapper` dictionary
        with the corresponding component class.

        :param components: List of names of the preprocessing components to include in this pipeline.
        :type components: List[str]
        :param config: YAML configuration for the pipeline.
        :type config: YAMLObject
        """
        for component in components:
            if component == 'file_mapper':
                from vanpy.core.preprocess_components.FilelistDataFrameCreator import FilelistDataFrameCreator
                self.components_mapper[component] = FilelistDataFrameCreator
            elif component == 'wav_converter':
                from vanpy.core.preprocess_components.WAVConverter import WAVConverter
                self.components_mapper[component] = WAVConverter
            elif component == 'wav_splitter':
                from vanpy.core.preprocess_components.WAVSplitter import WAVSplitter
                self.components_mapper[component] = WAVSplitter
            elif component == 'ina_speech_segmenter':
                from vanpy.core.preprocess_components.INAVoiceSeparator import INAVoiceSeparator
                self.components_mapper[component] = INAVoiceSeparator
            elif component == 'pyannote_vad':
                from vanpy.core.preprocess_components.PyannoteVAD import PyannoteVAD
                self.components_mapper[component] = PyannoteVAD
            elif component == 'silero_vad':
                from vanpy.core.preprocess_components.SileroVAD import SileroVAD
                self.components_mapper[component] = SileroVAD
            elif component == 'pyannote_sd':
                from vanpy.core.preprocess_components.PyannoteSD import PyannoteSD
                self.components_mapper[component] = PyannoteSD
            elif component == 'metricgan_se':
                from vanpy.core.preprocess_components.MetricGANSE import MetricGANSE
                self.components_mapper[component] = MetricGANSE
            elif component == 'sepformer_se':
                from vanpy.core.preprocess_components.SepFormerSE import SepFormerSE
                self.components_mapper[component] = SepFormerSE

        super().__init__(components, config)
        self.logger.info(f'Created Preprocessing Pipeline with {len(self.components)} components')
