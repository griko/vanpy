import subprocess

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.SegmenterComponent import SegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist
from yaml import YAMLObject
import pandas as pd


class WAVConverter(SegmenterComponent):
    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='wav_converter',
                         yaml_config=yaml_config)

    def run_ffmpeg(self, f, ab, ac, ar, output_dir, output_filename):
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                        f"{f}", "-ab", ab, "-ac", f"{ac}", "-ar", ar, f'{output_dir}/{output_filename}', '-dn',
                        '-ignore_unknown', '-sn'])

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        p_df = pd.DataFrame()
        processed_path, metadata = self.segmenter_create_columns(metadata)
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(p_df, paths_list, processed_path,
                                                                                   input_column, output_dir)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)

        ab = self.config['ab']
        ac = self.config['ac']
        ar = self.config['ar']
        for j, f in enumerate(paths_list):
            filename = ''.join(f.split("/")[-1].split(".")[:-1])
            if not output_dir:
                input_path = ''.join(f.split("/")[:-1])
                output_dir = input_path
            output_filename = f'{filename}.wav'
            self.run_ffmpeg(f, ab, ac, ar, output_dir, output_filename)

            f_df = pd.DataFrame.from_dict({processed_path: [f'{output_dir}/{output_filename}'],
                                           input_column: [f]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
            self.logger.info(f'Converted {f}, {j}/{len(paths_list)}')

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)

        return ComponentPayload(metadata=metadata, df=df)
