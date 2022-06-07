from yaml import YAMLObject
from pyannote.audio import Inference
import numpy as np
import pandas as pd

from audio_pipeline.core.PiplineComponent import PipelineComponent, ComponentPayload


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

        p_df = pd.DataFrame()
        for f in paths_list:
            try:
                embedding = self.model(f)
                f_df = pd.DataFrame(np.mean(embedding, axis=0)).T
                f_df[input_column] = f
                p_df = pd.concat([p_df, f_df], ignore_index=True)
                self.logger.info(f'done with {f}')
            except RuntimeError as e:
                self.logger.error(f'An error occurred in {f}: {e}')

        feature_columns = p_df.columns.tolist()
        feature_columns.remove(input_column)
        metadata['feature_columns'].extend(feature_columns)
        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
