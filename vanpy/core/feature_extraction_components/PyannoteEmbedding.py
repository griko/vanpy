import time

from yaml import YAMLObject
from pyannote.audio import Inference
from pyannote.audio import Model
import numpy as np
import pandas as pd

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PiplineComponent import PipelineComponent
from vanpy.utils.utils import get_null_wav_path


class PyannoteEmbedding(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='pyannote_embedding',
                         yaml_config=yaml_config)

    def load_model(self):
        model = Model.from_pretrained("pyannote%2Fembedding",
                                      use_auth_token="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")
        self.model = Inference(model,
                               window="sliding",
                               duration=self.config['sliding_window_duration'],
                               step=self.config['sliding_window_step'])

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        df = df.reset_index().drop(['index'], axis=1, errors='ignore')
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()

        file_performance_column_name = ''
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_features'
            metadata['meta_columns'].extend([file_performance_column_name])
            if file_performance_column_name in df.columns:
                df = df.drop([file_performance_column_name], axis=1)
            df.insert(0, file_performance_column_name, None)

        # replace the feature columns
        embedding = self.model(get_null_wav_path())
        f_df = pd.DataFrame(np.mean(embedding, axis=0)).T
        for c in f_df.columns[::-1]:
            c = f'{c}_{self.get_name()}'
            if c in df.columns:
                df = df.drop([c], axis=1)
            df.insert(0, c, None)
        feature_columns = f_df.columns.tolist()

        for j, f in enumerate(paths_list):
            try:
                t_start_feature_extraction = time.time()
                embedding = self.model(f)
                f_df = pd.DataFrame(np.mean(embedding, axis=0)).T
                f_df[input_column] = f
                t_end_feature_extraction = time.time()
                if self.config['performance_measurement']:
                    f_df[file_performance_column_name] = t_end_feature_extraction - t_start_feature_extraction
                for c in f_df.columns:
                    df.at[j, f'{c}_{self.get_name()}'] = f_df.iloc[0, f_df.columns.get_loc(c)]
                self.logger.info(f'done with {f}, {j}/{len(paths_list)}')
            except RuntimeError as e:
                self.logger.error(f'An error occurred in {f}, {j}/{len(paths_list)}: {e}')
            self.save_intermediate_payload(j, ComponentPayload(metadata=metadata, df=df))

        metadata['feature_columns'].extend(feature_columns)
        return ComponentPayload(metadata=metadata, df=df)
