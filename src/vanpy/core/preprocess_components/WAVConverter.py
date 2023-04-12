import subprocess

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.SegmenterComponent import SegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist
from yaml import YAMLObject
import pandas as pd


class WAVConverter(SegmenterComponent):
    """
    A preprocessing component to convert audio files to WAV format using FFMPEG.
    """
    def __init__(self, yaml_config: YAMLObject):
        """
        Initializes the WAVConverter class and creates initial ffmpeg configuration parameters
        :param yaml_config: A YAMLObject containing the configuration for the pipeline
        """
        super().__init__(component_type='preprocessing', component_name='wav_converter',
                         yaml_config=yaml_config)
        self.ffmpeg_config = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                              "input_file", "-vn", "output_file", '-dn', '-ignore_unknown', '-sn']

    def update_ffmpeg_config(self):
        """
        Update the FFMPEG configuration with available parameters.

        :return: None
        """
        available_parameters = ['ab', 'ac', 'ar', 'acodec']
        input_file_idx = self.ffmpeg_config.index("input_file")
        for ap in available_parameters:
            if ap in self.config:
                self.ffmpeg_config.insert(input_file_idx + 1, str(self.config[ap]))
                self.ffmpeg_config.insert(input_file_idx + 1, "-" + ap)

    def run_ffmpeg(self, f: str, output_dir: str, output_filename: str):
        """
        Run FFMPEG to convert an audio file to WAV format.

        :param f: The path of the input audio file.
        :param output_dir: The path of the output directory.
        :param output_filename: The name of the output WAV file.
        :return: None
        """
        ffmpeg_config = self.ffmpeg_config.copy()
        input_file_idx = ffmpeg_config.index("input_file")
        output_file_idx = ffmpeg_config.index("output_file")
        ffmpeg_config[input_file_idx] = f"{f}"
        ffmpeg_config[output_file_idx] = f'{output_dir}/{output_filename}'

        subprocess.run(ffmpeg_config)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Convert audio files to WAV format using FFMPEG.

        :param input_payload: A ComponentPayload object containing metadata and a DataFrame with audio file paths.
        :return: A ComponentPayload object containing metadata and a DataFrame with the converted audio file paths.
        """
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
                                                                                   use_dir_prefix=self.config.get('use_dir_name_as_prefix', False))

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)
        self.config['records_count'] = len(paths_list)

        self.update_ffmpeg_config()

        for j, f in enumerate(paths_list):
            filename = ''.join(f.split("/")[-1].split(".")[:-1])
            dir_prefix = ''
            if self.config.get('use_dir_name_as_prefix', False):
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
