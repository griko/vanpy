from yaml import YAMLObject

from core.PipelinePreprocessingComponent import PipelinePreprocessingComponent
from core.PiplineComponent import PipelineComponent, ComponentPayload
from utils.utils import create_dirs_if_not_exist, cut_segment
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

    def process(self, input_object: ComponentPayload) -> ComponentPayload:
        features, df = input_object.unpack()
        input_column = features['paths_column']
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        filtered_dir = self.config['filtered_dir']
        create_dirs_if_not_exist(output_dir, filtered_dir)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_object

        p_df = pd.DataFrame()
        processed_path = f'{self.get_name()}_processed_path'
        features['paths_column'] = processed_path
        for f in paths_list:
            try:
                start = time.time()
                segmentation = self.seg(f)
                v_segments, f_segments = INAVoiceSeparator.get_voice_segments(segmentation)
                for i, segment in enumerate(v_segments):
                    output_path = cut_segment(f, output_dir=output_dir, segment=segment, segment_id=i)
                    f_df = pd.DataFrame.from_dict({processed_path: [output_path],
                                                   f'{self.get_name()}_segment_start': [segment[0]],
                                                   f'{self.get_name()}_segment_stop': [segment[1]],
                                                   input_column: [f]})
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
                end = time.time()
                self.logger.info(f'Extracted {len(v_segments)} from {f} in {end - start} seconds')

            except AssertionError as err:
                self.logger.error(f"Error reading {f}")

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(features=features, df=df)

