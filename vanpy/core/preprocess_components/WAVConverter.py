import subprocess

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PiplineComponent import PipelineComponent
from vanpy.utils.utils import create_dirs_if_not_exist
from yaml import YAMLObject
import pandas as pd


class WAVConverter(PipelineComponent):
    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='wav_converter',
                         yaml_config=yaml_config)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to convert')
            return input_payload

        p_df = pd.DataFrame()
        processed_path = f'{self.get_name()}_processed_path'
        metadata['paths_column'] = processed_path
        metadata['all_paths_columns'].append(processed_path)

        ab = self.config['ab']
        ac = self.config['ac']
        ar = self.config['ar']
        for f in paths_list:
            filename = ''.join(f.split("/")[-1].split(".")[:-1])
            if not output_dir:
                input_path = ''.join(f.split("/")[:-1])
                output_dir = input_path
            output_filename = f'{filename}.wav'

            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                            f"{f}", "-ab", ab, "-ac", f"{ac}", "-ar", ar, f'{output_dir}/{output_filename}', '-dn',
                            '-ignore_unknown', '-sn'])

            f_df = pd.DataFrame.from_dict({processed_path: [f'{output_dir}/{output_filename}'],
                                           input_column: [f]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
            self.logger.info(f'done with {f}')

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)

        return ComponentPayload(metadata=metadata, df=df)
