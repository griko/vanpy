from yaml import YAMLObject

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment
from inaSpeechSegmenter import Segmenter
import pandas as pd
import time


class INAVoiceSeparator(BaseSegmenterComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='ina_speech_segmenter',
                         yaml_config=yaml_config)

    def load_model(self):
        self.model = Segmenter(vad_engine=self.config['vad_engine'])

    @staticmethod
    def get_voice_segments(segmentation):
        voice_sections, filtered_sections = [], []
        for s in segmentation:
            kind, start, stop = s
            if kind == 'female' or kind == 'male':
                voice_sections.append((start, stop))
            else:
                filtered_sections.append((start, stop))
        return voice_sections, filtered_sections

    def process_item(self, f, p_df, processed_path, input_column, output_dir):
        t_start_segmentation = time.time()
        segmentation = self.model(f)
        v_segments, f_segments = INAVoiceSeparator.get_voice_segments(segmentation)
        t_end_segmentation = time.time()
        for i, segment in enumerate(v_segments):
            output_path = cut_segment(f, output_dir=output_dir, segment=segment, segment_id=i,
                                      separator=self.segment_name_separator, keep_only_first_segment=True)
            f_d = {processed_path: [output_path], input_column: [f]}
            self.add_segment_metadata(f_d, segment[0], segment[1])
            self.add_performance_metadata(f_d, t_start_segmentation, t_end_segmentation)
            f_df = pd.DataFrame.from_dict(f_d)
            p_df = pd.concat([p_df, f_df], ignore_index=True)
        return p_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        processed_path = self.get_processed_path()
        p_df = pd.DataFrame()
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(p_df, paths_list, processed_path,
                                                                                   input_column, output_dir)
        metadata = self.enhance_metadata(metadata)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)
        self.config['records_count'] = len(paths_list)

        p_df = self.process_with_progress(paths_list, metadata, self.process_item, p_df, processed_path, input_column, output_dir)

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
