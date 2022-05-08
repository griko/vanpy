import subprocess
from utils.utils import get_audio_files_paths, create_dirs_if_not_exist
from core.PiplineComponent import PipelineComponent
from yaml import YAMLObject


class WAVConverter(PipelineComponent):

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='encoding_converter', component_name='wav_converter',
                         yaml_config=yaml_config)

    def process(self) -> str:
        input_path = self.config['input_dir']
        paths_list = get_audio_files_paths(input_path)
        output_path = self.config['output_dir']
        create_dirs_if_not_exist(output_path)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to convert')
            return input_path

        for f in paths_list:
            filename = ''.join(f.split("/")[-1].split(".")[:-1])
            if not output_path:
                input_path = ''.join(f.split("/")[:-1])
                output_path = input_path
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                            f"{f}", "-ab", "256k", "-ac", "1", "-ar", "16k", f'{output_path}/{filename}.wav', '-dn',
                            '-ignore_unknown', '-sn'])
        return output_path
