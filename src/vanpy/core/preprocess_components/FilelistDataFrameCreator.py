import pandas as pd

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import get_audio_files_paths
from yaml import YAMLObject
import pickle


class FilelistDataFrameCreator(PipelineComponent):
    """
    A `PipelineComponent` that creates a DataFrame containing file paths of audio files within a given input_path. Or
    takes an existent ComponentPayload by reading the csv mentioned in the configuration yaml.
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
        Process the input_payload by creating a DataFrame containing file paths of audio files or loads an
        existing DataFrame from csv, configuring metadata path for further components that pick up the returned
        ComponentPayload
        :param input_payload: A `ComponentPayload` object containing the input data and input_path in metadata.
        :return: A modified `ComponentPayload` object containing the new DataFrame.
        """
        metadata, df = input_payload.unpack()

        if self.config.get('load_payload', False):
            p_df = pd.read_csv(self.config['load_df_path'])
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
