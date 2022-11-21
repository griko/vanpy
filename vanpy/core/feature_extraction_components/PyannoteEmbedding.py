import time

from yaml import YAMLObject
from pyannote.audio import Inference
import numpy as np
import pandas as pd

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PiplineComponent import PipelineComponent


class PyannoteEmbedding(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='pyannote_embedding',
                         yaml_config=yaml_config)

    def load_model(self):
        self.model = Inference("pyannote%2Fembedding",
                               window="sliding",
                               duration=self.config['sliding_window_duration'],
                               step=self.config['sliding_window_step'])

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()

        file_performance_column_name = ''
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_features'
            metadata['meta_columns'].extend([file_performance_column_name])
        p_df = pd.DataFrame()
        for j, f in enumerate(paths_list):
            try:
                t_start_feature_extraction = time.time()
                embedding = self.model(f)
                f_df = pd.DataFrame(np.mean(embedding, axis=0)).T
                f_df[input_column] = f
                t_end_feature_extraction = time.time()
                if self.config['performance_measurement']:
                    f_df[file_performance_column_name] = t_end_feature_extraction - t_start_feature_extraction
                p_df = pd.concat([p_df, f_df], ignore_index=True)
                self.logger.info(f'done with {f}, {j}/{len(paths_list)}')
            except RuntimeError as e:
                self.logger.error(f'An error occurred in {f}, {j}/{len(paths_list)}: {e}')

        feature_columns = p_df.columns.tolist()
        feature_columns.remove(input_column)
        metadata['feature_columns'].extend(feature_columns)
        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
