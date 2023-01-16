from typing import List
from yaml import YAMLObject
from vanpy.core.BasePipline import BasePipeline
# from vanpy.core.preprocess_components.ESPnetSpeechEnhancement import ESPnetSpeechEnhancement
from vanpy.core.preprocess_components.PyannoteVAD import PyannoteVAD
from vanpy.core.preprocess_components.SileroVAD import SileroVAD
from vanpy.core.preprocess_components.WAVConverter import WAVConverter
from vanpy.core.preprocess_components.INAVoiceSeparator import INAVoiceSeparator
from vanpy.core.preprocess_components.FilelistDataFrameCreator import FilelistDataFrameCreator


class PreprocessPipeline(BasePipeline):
    components_mapper = {
        'file_mapper': FilelistDataFrameCreator,
        'wav_converter': WAVConverter,
        'ina_speech_segmenter': INAVoiceSeparator,
        'pyannote_vad': PyannoteVAD,
        'silero_vad': SileroVAD,
        # 'espnet-se': ESPnetSpeechEnhancement
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Preprocessing Pipeline with {len(self.components)} components')
