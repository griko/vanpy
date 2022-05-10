from typing import List, Tuple
import pandas as pd

from yaml import YAMLObject

from core.BasePipline import BasePipeline
from core.PipelinePreprocessingComponent import PipelinePreprocessingComponent
from core.preprocess_components.embedding.PyannoteEmbedding import PyannoteEmbedding
from core.preprocess_components.encoding_converter.WAVConverter import WAVConverter
from core.preprocess_components.voice_music_separator.INAVoiceSeparator import INAVoiceSeparator


class PreprocessPipeline(BasePipeline):
    components: List[PipelinePreprocessingComponent]
    components_mapper = {
        'wav_converter': WAVConverter,
        'ina_speech_segmenter': INAVoiceSeparator,
        'pyannote_embedding': PyannoteEmbedding
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Preprocessing Pipeline with {len(self.components)} components')

    def process(self) -> Tuple[str, pd.DataFrame]:
        process_dir = self.components[0].get_input_dir()
        process_df = pd.DataFrame()
        for component in self.components:
            self.logger.info(f'Processing with {component.get_name()}')
            process_dir, process_df = component.process(process_dir, process_df)
        return process_dir, process_df
