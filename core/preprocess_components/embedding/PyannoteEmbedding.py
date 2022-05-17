from yaml import YAMLObject
from typing import Tuple

from core.PipelinePreprocessingComponent import PipelinePreprocessingComponent
from pyannote.audio import Inference
import numpy as np
import pandas as pd

from core.PiplineComponent import PipelineComponent, ComponentPayload
from utils.utils import get_audio_files_paths


class PyannoteEmbedding(PipelineComponent):
    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='embedding', component_name='pyannote_embedding',
                         yaml_config=yaml_config)
        self.inference_emb = Inference("pyannote%2Fembedding",
                                  window="sliding",
                                  duration=self.config['sliding_window_duration'], step=self.config['sliding_window_step'])

    def process(self, input_object: ComponentPayload) -> ComponentPayload:
        features, df = input_object.unpack()
        input_column = features['paths_column']
        paths_list = df[input_column].tolist()

        p_df = pd.DataFrame()
        for f in paths_list:
            try:
                embedding = self.inference_emb(f)
                f_df = pd.DataFrame(np.mean(embedding, axis=0)).T
                f_df[input_column] = f
                p_df = pd.concat([p_df, f_df], ignore_index=True)
                self.logger.info(f'done with {f}')
            except RuntimeError as e:
                self.logger.error(f'An error occurred in {f}: {e}')

        embedding_columns = p_df.columns.tolist()
        embedding_columns.remove(input_column)
        features['embedding_columns'] = embedding_columns
        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(features=features, df=df)
        # df_pyannote = df
        # X_test_pyannote = df_pyannote.drop(['filename'], axis=1)
        # y_test_pyannote = df_pyannote[['filename']]
