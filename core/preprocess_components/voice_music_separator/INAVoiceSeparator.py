from yaml import YAMLObject

from core.PipelinePreprocessingComponent import PipelinePreprocessingComponent
from core.PiplineComponent import PipelineComponent
from utils.utils import get_audio_files_paths, create_dirs_if_not_exist, cut_by_segments
from inaSpeechSegmenter import Segmenter
from typing import Tuple
import pandas as pd
import time


class INAVoiceSeparator(PipelineComponent):
    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='voice_music_separator', component_name='ina_speech_segmenter',
                         yaml_config=yaml_config)
        self.seg = Segmenter(vad_engine=self.config['vad_engine'])

    @classmethod
    def get_voice_segments(cls, segmentation):
        voice_sections, filtered_sections = [], []
        for s in segmentation:
            kind, start, stop = s
            if kind == 'female' or kind == 'male':
                voice_sections.append((start, stop))
            else:
                filtered_sections.append((start, stop))
        return voice_sections, filtered_sections

    def process(self, input_dir: str = '', df: pd.DataFrame = None) -> Tuple[str, pd.DataFrame]:
        paths_list = get_audio_files_paths(input_dir, extension='.wav')
        output_path = self.config['output_dir']
        filtered_path = self.config['filtered_dir']
        create_dirs_if_not_exist(output_path, filtered_path)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_dir, df

        for f in paths_list:
            try:
                start = time.time()
                segmentation = self.seg(f)
                v_segments, f_segments = INAVoiceSeparator.get_voice_segments(segmentation)
                cut_by_segments(f, output_path, v_segments)
                cut_by_segments(f, filtered_path, f_segments)
                end = time.time()
                self.logger.info(f'Extracted {len(v_segments)} from {f} in {end - start} seconds')
            except AssertionError as err:
                self.logger.error(f"Error reading {f}")

        return output_path, df
