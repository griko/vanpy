import pandas as pd

from src.vanpy.core.ComponentPayload import ComponentPayload
from src.vanpy.core.PipelineComponent import PipelineComponent
from src.vanpy.utils.utils import get_audio_files_paths
from yaml import YAMLObject
import pickle


class FilelistDataFrameCreator(PipelineComponent):
    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='file_mapper',
                         yaml_config=yaml_config)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        metadata, df = input_payload.unpack()
        if 'input_path' not in metadata:
            raise AttributeError("The supplied ComponentPayload does not contain 'input_path', file_mapper can not be used without it")
        input_folder = metadata['input_path']
        if 'load_payload' in self.config and self.config['load_payload']:
            p_df = pd.read_csv(self.config['load_df_path'])
            with open(self.config['load_meta_path'], 'rb') as pickle_file:
                metadata = pickle.load(pickle_file)
        else:
            paths_list = get_audio_files_paths(input_folder)
            processed_path = f'{self.component_name}_paths'
            metadata['paths_column'] = processed_path
            metadata['all_paths_columns'].append(processed_path)
            p_df = pd.DataFrame(paths_list, columns=[processed_path])
        return ComponentPayload(metadata=metadata, df=p_df)
