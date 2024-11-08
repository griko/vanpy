from abc import ABC
from typing import Dict, List, Tuple, Union
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import get_audio_files_paths
import pandas as pd


class BaseSegmenterComponent(PipelineComponent, ABC):
    """
    Base class for `PipelineComponent`s for segmenting audio files into smaller segments.
    """

    def __init__(self, component_type: str, component_name: str, yaml_config: YAMLObject):
        """
        Initializes a new instance of the SegmenterComponent class.

        :param component_type: The type of pipeline component.
        :param component_name: The name of the pipeline component.
        :param yaml_config: The YAML configuration object.
        """
        super().__init__(component_type, component_name, yaml_config)
        self.segment_name_separator = self.config.get('segment_name_separator', '_')
        self.segment_stop_column_name = None
        self.segment_start_column_name = None
        self.classification_column_name = None

    def get_processed_path(self):
        return f'{self.get_name()}_processed_path'

    def add_segment_columns_to_metadata(self, metadata: Dict) -> Dict:
        """
        Creates the columns for ComponentPayload for the segmented audio files.
        """
        self.segment_start_column_name = self.segment_stop_column_name = ''
        if self.config.get('add_segment_metadata', True):
            self.segment_start_column_name = f'{self.get_name()}_segment_start'
            self.segment_stop_column_name = f'{self.get_name()}_segment_stop'
            metadata['meta_columns'].extend(
                [self.segment_start_column_name, self.segment_stop_column_name])
        self.file_performance_column_name = ''
        return metadata

    def add_processed_path_to_metadata(self, processed_path, metadata: Dict) -> Dict:
        metadata['paths_column'] = processed_path
        metadata['all_paths_columns'].append(processed_path)
        return metadata

    def add_classification_column_to_metadata(self, metadata: Dict) -> Dict:
        metadata['classification_columns'].extend([self.classification_column_name])
        return metadata

    def enhance_metadata(self, metadata: Dict) -> Dict:
        metadata = self.add_segment_columns_to_metadata(metadata)
        metadata = self.add_performance_column_to_metadata(metadata)
        metadata = self.add_processed_path_to_metadata(self.get_processed_path(), metadata)
        if self.classification_column_name is not None:
            metadata = self.add_classification_column_to_metadata(metadata)
        return metadata

    def add_segment_metadata(self, f_d: Union[pd.DataFrame, Dict], a: float, b: float):
        """
        Adds metadata for the start and stop time of audio segments to the temporal DataFrame.

        :param f_d: The temporal DataFrame that is enhanced with the start and stop time of audio segments.
        :param a: The start time of the audio segment in the input audio file.
        :param b: The stop time of the audio segment in the input audio file.
        """
        if self.config.get('add_segment_metadata', True):
            f_d[self.segment_start_column_name] = [a]
            f_d[self.segment_stop_column_name] = [b]

    def get_file_paths_and_processed_df_if_not_overwriting(self, paths_list: List[str],
                                                           processed_path: str, input_column: str,
                                                           output_dir: str, use_dir_prefix: bool = False) -> Tuple[
        pd.DataFrame, List[str]]:
        """
        Returns a processed DataFrame and a list of unprocessed file paths if not overwriting existing files.

        :param p_df: A pandas DataFrame to append processed file paths and input file paths to.
        :param paths_list: A list of file paths to process.
        :param processed_path: The column name to append the processed file path to the `p_df`.
        :param input_column: The column name to append the input file path to the `p_df`.
        :param output_dir: The output directory path.
        :param use_dir_prefix: If True, the directory name is added as a prefix to the processed file name.
        :return: A tuple containing the updated `p_df` DataFrame and a list of unprocessed file paths.
        """
        unprocessed_paths_list = []
        p_df = pd.DataFrame()
        if not self.config.get('overwrite', False):
            existing_file_list = get_audio_files_paths(output_dir)
            existing_file_list_names = ['.'.join(f.split("/")[-1].split('.')[0:-1]) for f in existing_file_list]
            existing_file_set = {}
            for p in existing_file_list:
                short_name = f'{self.segment_name_separator}'.join('.'.join(p.split("/")[-1].split(".")[0:-1])
                                                                   .split(f'{self.segment_name_separator}')[:-1])
                if short_name in existing_file_set:
                    existing_file_set[short_name].append(p)
                else:
                    existing_file_set[short_name] = [p]

            for f in paths_list:
                file_name_without_extension = f.split("/")[-1].split(".")[0]
                if use_dir_prefix:
                    file_name_without_extension = f.split("/")[-2] + '_' + file_name_without_extension
                if file_name_without_extension in existing_file_set:
                    f_df = pd.DataFrame.from_dict(
                        {processed_path: [existing_file_set[file_name_without_extension]], input_column: [f]})
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
                elif file_name_without_extension in existing_file_list_names:
                    f_df = pd.DataFrame.from_dict(
                        {processed_path: [
                            existing_file_list[existing_file_list_names.index(file_name_without_extension)]],
                         input_column: [f]})
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
                else:
                    unprocessed_paths_list.append(f)
            if processed_path in p_df:
                p_df = p_df.explode(processed_path).reset_index().drop(['index'], axis=1)
        else:
            unprocessed_paths_list = paths_list
        for col in [processed_path, input_column]:  # add required columns to pass merge
            if col not in p_df:
                p_df[col] = None
        return p_df, unprocessed_paths_list
