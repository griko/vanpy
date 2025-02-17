import pandas as pd

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import get_audio_files_paths
from yaml import YAMLObject
import pickle


class FilelistDataFrameCreator(PipelineComponent):
    """
    Component for creating a DataFrame of audio file paths.

    Creates a DataFrame containing paths to audio files from either:
    - A directory specified in the input path
    - An existing CSV file specified in configuration

    :ivar config: Configuration dictionary containing load/save paths.
    """
    def __init__(self, yaml_config: YAMLObject):
        """
        Initializes the FilelistDataFrameCreator class
        :param yaml_config: A YAMLObject containing the configuration for the pipeline
        """
        super().__init__(component_type='preprocessing', component_name='file_mapper',
                         yaml_config=yaml_config)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Create a DataFrame of audio file paths.

        Either loads paths from an existing CSV or scans a directory for audio files.
        Updates metadata with path column information.

        :param input_payload: Input payload containing path information.
        :return: Output payload containing file path DataFrame.
        :raises AttributeError: If neither input_path nor load_payload is provided.
        """
        metadata, df = input_payload.unpack()

        if self.config.get('load_payload', False):
            p_df = pd.read_csv(self.config['load_df_path'])
            if 'load_meta_path' in self.config:
                with open(self.config['load_meta_path'], 'rb') as pickle_file:
                    metadata = pickle.load(pickle_file)
        else:
            if 'input_path' not in metadata:
                raise AttributeError(
                    "The supplied ComponentPayload neither contain 'input_path' nor 'load_payload' is set, file_mapper can not be used without it")
            input_folder = metadata['input_path']
            paths_list = get_audio_files_paths(input_folder)
            processed_path = f'{self.component_name}_paths'
            metadata['paths_column'] = processed_path
            metadata['all_paths_columns'].append(processed_path)
            p_df = pd.DataFrame(paths_list, columns=[processed_path])
        return ComponentPayload(metadata=metadata, df=p_df)
