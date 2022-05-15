from yaml import YAMLObject
from typing import Tuple

from core.PipelinePreprocessingComponent import PipelinePreprocessingComponent
from pyannote.audio import Inference
import numpy as np
import pandas as pd

from core.PiplineComponent import PipelineComponent
from utils.utils import get_audio_files_paths


class PyannoteEmbedding(PipelineComponent):
    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='embedding', component_name='pyannote_embedding',
                         yaml_config=yaml_config)
        self.inference_emb = Inference("pyannote%2Fembedding",
                                  window="sliding",
                                  duration=self.config['sliding_window_duration'], step=self.config['sliding_window_step'])

    def process(self, input_dir: str = '', df: pd.DataFrame = None) -> Tuple[str, pd.DataFrame]:
        paths_list = get_audio_files_paths(input_dir, '.wav')

        for f in paths_list:
            try:
                embedding = self.inference_emb(f)
                tmp = pd.DataFrame(np.mean(embedding, axis=0)).T
                tmp['filename'] = f.split("/")[-1]
                df = pd.concat([tmp, df], ignore_index=True)
                self.logger.info(f'done with {f}')
            except RuntimeError as e:
                self.logger.error(f'An error occurred in {f}: {e}')
        return input_dir, df
        # df_pyannote = df
        # X_test_pyannote = df_pyannote.drop(['filename'], axis=1)
        # y_test_pyannote = df_pyannote[['filename']]
