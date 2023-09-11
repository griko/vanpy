import subprocess

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist
from yaml import YAMLObject
import pandas as pd
from pydub import AudioSegment
from tqdm.auto import tqdm


class WAVConverter(BaseSegmenterComponent):
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
        self.params_list = self.get_parameters_from_config()

    def get_parameters_from_config(self):
        params_list =[]
        available_parameters = {'ab': "256k", 'ac': "1", 'ar': "16000", 'acodec': "pcm_s16le"}
        for ap in available_parameters.keys():
            params_list.append("-" + ap)
            if ap in self.config:
                params_list.append(str(self.config[ap]))
            else:
                params_list.append(available_parameters[ap])
        return params_list

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
        processed_path = self.get_processed_path()
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(p_df, paths_list, processed_path,
                                                                                   input_column, output_dir,
                                                                                   use_dir_prefix=self.config.get('use_dir_name_as_prefix', False))
        metadata = self.enhance_metadata(metadata)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)
        self.config['records_count'] = len(paths_list)

        for j, f in enumerate(tqdm(paths_list)):
            filename = ''.join(f.split("/")[-1].split(".")[:-1])
            dir_prefix = ''
            if self.config.get('use_dir_name_as_prefix', False):
                dir_prefix = f.split("/")[-2] + '_'
            if not output_dir:
                input_path = ''.join(f.split("/")[:-1])
                output_dir = input_path
            output_filename = f'{dir_prefix}{filename}.wav'
            AudioSegment.from_file(f).export(f'{output_dir}/{output_filename}', format='wav', parameters=self.params_list)

            f_df = pd.DataFrame.from_dict({processed_path: [f'{output_dir}/{output_filename}'],
                                           input_column: [f]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
            self.latent_info_log(f'Converted {f}, {j + 1}/{len(paths_list)}', iteration=j)
        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)

        return ComponentPayload(metadata=metadata, df=df)
