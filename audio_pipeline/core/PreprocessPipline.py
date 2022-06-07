from typing import List
from yaml import YAMLObject
from audio_pipeline.core.BasePipline import BasePipeline
from audio_pipeline.core.preprocess_components.PyannoteVAD import PyannoteVAD
from audio_pipeline.core.preprocess_components.WAVConverter import WAVConverter
from audio_pipeline.core.preprocess_components.INAVoiceSeparator import INAVoiceSeparator
from audio_pipeline.core.preprocess_components.FilelistDataFrameCreator import FilelistDataFrameCreator


class PreprocessPipeline(BasePipeline):
    components_mapper = {
        'file_mapper': FilelistDataFrameCreator,
        'wav_converter': WAVConverter,
        'ina_speech_segmenter': INAVoiceSeparator,
        'pyannote_vad': PyannoteVAD
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Preprocessing Pipeline with {len(self.components)} components')
