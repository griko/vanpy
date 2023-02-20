from typing import List
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline

class PreprocessPipeline(BasePipeline):
    components_mapper = {
        'file_mapper': None,
        'wav_converter': None,
        'ina_speech_segmenter': None,
        'pyannote_vad': None,
        'pyannote_sd': None,
        'silero_vad': None,
        'metricgan_se': None,
    }

    def __init__(self, components: List[str], config: YAMLObject):
        for component in components:
            if component == 'file_mapper':
                from vanpy.core.preprocess_components.FilelistDataFrameCreator import FilelistDataFrameCreator
                self.components_mapper[component] = FilelistDataFrameCreator
            elif component == 'wav_converter':
                from vanpy.core.preprocess_components.WAVConverter import WAVConverter
                self.components_mapper[component] = WAVConverter
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

        super().__init__(components, config)
        self.logger.info(f'Created Preprocessing Pipeline with {len(self.components)} components')
