import subprocess

from src.vanpy.core.ComponentPayload import ComponentPayload
from src.vanpy.core.preprocess_components.SegmenterComponent import SegmenterComponent
from src.vanpy.utils.utils import create_dirs_if_not_exist
from yaml import YAMLObject
import pandas as pd


class WAVConverter(SegmenterComponent):
    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='wav_converter',
                         yaml_config=yaml_config)
        self.ffmpeg_config = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                              "input_file", "-vn", "output_file", '-dn', '-ignore_unknown', '-sn']

    def update_ffmpeg_config(self):
        available_parameters = ['ab', 'ac', 'ar', 'acodec']
        input_file_idx = self.ffmpeg_config.index("input_file")
        for ap in available_parameters:
            if ap in self.config:
                self.ffmpeg_config.insert(input_file_idx + 1, str(self.config[ap]))
                self.ffmpeg_config.insert(input_file_idx + 1, "-" + ap)


    def run_ffmpeg(self, f, output_dir, output_filename):
        ffmpeg_config = self.ffmpeg_config.copy()
        input_file_idx = ffmpeg_config.index("input_file")
        output_file_idx = ffmpeg_config.index("output_file")
        ffmpeg_config[input_file_idx] = f"{f}"
        ffmpeg_config[output_file_idx] = f'{output_dir}/{output_filename}'

        subprocess.run(ffmpeg_config)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        if input_column == '':
            raise KeyError("WAV converter can not run without specifying a paths column in the payload. Maybe you should run the file_maper before.")
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        p_df = pd.DataFrame()
        processed_path, metadata = self.segmenter_create_columns(metadata)
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(p_df, paths_list, processed_path,
                                                                                   input_column, output_dir,
                                                                                   use_dir_prefix='use_dir_name_as_prefix' in self.config and self.config['use_dir_name_as_prefix'])

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)
        self.config['items_in_paths_list'] = len(paths_list) - 1

        self.update_ffmpeg_config()

        for j, f in enumerate(paths_list):
            filename = ''.join(f.split("/")[-1].split(".")[:-1])
            dir_prefix = ''
            if 'use_dir_name_as_prefix' in self.config and self.config['use_dir_name_as_prefix']:
                dir_prefix = f.split("/")[-2] + '_'
            if not output_dir:
                input_path = ''.join(f.split("/")[:-1])
                output_dir = input_path
            output_filename = f'{dir_prefix}{filename}.wav'
            self.run_ffmpeg(f, output_dir, output_filename)

            f_df = pd.DataFrame.from_dict({processed_path: [f'{output_dir}/{output_filename}'],
                                           input_column: [f]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
            self.latent_info_log(f'Converted {f}, {j + 1}/{len(paths_list)}', iteration=j)
        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)

        return ComponentPayload(metadata=metadata, df=df)
