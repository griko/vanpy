from abc import ABC

from yaml import YAMLObject

from vanpy.core.PiplineComponent import PipelineComponent
from vanpy.utils.utils import get_audio_files_paths
import pandas as pd


class SegmenterComponent(PipelineComponent, ABC):
    def __init__(self, component_type: str, component_name: str, yaml_config: YAMLObject):
        super().__init__(component_type, component_name, yaml_config)
        self.segment_name_separator = yaml_config['segment_name_separator']
        self.segment_stop_column_name = None
        self.segment_start_column_name = None
        self.file_performance_column_name = None

    def segmenter_create_columns(self, metadata):
        processed_path = f'{self.get_name()}_processed_path'
        metadata['paths_column'] = processed_path
        metadata['all_paths_columns'].append(processed_path)
        self.segment_start_column_name = self.segment_stop_column_name = ''
        if self.config['add_segment_metadata']:
            self.segment_start_column_name = f'{self.get_name()}_segment_start'
            self.segment_stop_column_name = f'{self.get_name()}_segment_stop'
            metadata['meta_columns'].extend([self.segment_start_column_name, self.segment_stop_column_name])
        self.file_performance_column_name = ''
        if self.config['performance_measurement']:
            self.file_performance_column_name = f'perf_{self.get_name()}_get_voice_segments'
            metadata['meta_columns'].extend([self.file_performance_column_name])
        return processed_path, metadata

    def add_segment_metadata(self, f_d, a, b):
        if self.config['add_segment_metadata']:
            f_d[self.segment_start_column_name] = [a]
            f_d[self.segment_stop_column_name] = [b]

    def add_performance_metadata(self, f_d, t_start, t_end):
        if self.config['performance_measurement']:
            f_d[self.file_performance_column_name] = t_end - t_start

    def get_file_paths_and_processed_df_if_not_overwriting(self, p_df, paths_list, processed_path, input_column,
                                                           output_dir):
        unprocessed_paths_list = []
        if not self.config['overwrite']:
            existing_file_list = get_audio_files_paths(output_dir)
            existing_file_set = {f'{self.segment_name_separator}'
                                     .join('.'.join(p.split("/")[-1].split(".")[0:-1])
                                           .split(f'{self.segment_name_separator}')[:-1]): p for p in
                                 existing_file_list}
            for f in paths_list:
                file_name_without_extension = f.split("/")[-1].split(".")[0]
                if file_name_without_extension in existing_file_set:
                    f_df = pd.DataFrame.from_dict(
                        {processed_path: [existing_file_set[file_name_without_extension]], input_column: [f]})
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
                else:
                    unprocessed_paths_list.append(f)
        else:
            unprocessed_paths_list = paths_list
        return p_df, unprocessed_paths_list
