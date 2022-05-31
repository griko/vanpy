from typing import List
from yaml import YAMLObject
from core.BasePipline import BasePipeline
from core.preprocess_components.encoding_converter.WAVConverter import WAVConverter
from core.preprocess_components.voice_music_separator.INAVoiceSeparator import INAVoiceSeparator
from core.preprocess_components.file_mapper.FilelistDataFrameCreator import FilelistDataFrameCreator


class PreprocessPipeline(BasePipeline):
    components_mapper = {
        'file_mapper': FilelistDataFrameCreator,
        'wav_converter': WAVConverter,
        'ina_speech_segmenter': INAVoiceSeparator
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Preprocessing Pipeline with {len(self.components)} components')
