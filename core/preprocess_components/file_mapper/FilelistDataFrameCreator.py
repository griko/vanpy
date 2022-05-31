import pandas as pd

from core.PiplineComponent import PipelineComponent, ComponentPayload
from utils.utils import get_audio_files_paths
from yaml import YAMLObject


class FilelistDataFrameCreator(PipelineComponent):
    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='file_mapper',
                         yaml_config=yaml_config)

    def process(self, input_object: ComponentPayload) -> ComponentPayload:
        features, df = input_object.unpack()
        input_folder = features['input_path']
        paths_list = get_audio_files_paths(input_folder)
        processed_path = f'{self.component_name}_paths'
        features['paths_column'] = processed_path
        p_df = pd.DataFrame()
        for f in paths_list:
            f_df = pd.DataFrame.from_dict({processed_path: [f]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
        return ComponentPayload(features=features, df=p_df)
