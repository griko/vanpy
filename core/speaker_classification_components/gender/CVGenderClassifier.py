from yaml import YAMLObject

from core.PiplineComponent import PipelineComponent
from pyannote.audio import Inference
import numpy as np
import pandas as pd

from utils.utils import get_audio_files_paths


class CVGenderClassifier(PipelineComponent):

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='speaker_classifier', component_name='common_voices_gender',
                         yaml_config=yaml_config)
        self.inference_emb = Inference("pyannote%2Fembedding",
                                  window="sliding",
                                  duration=self.config['sliding_window_duration'], step=self.config['sliding_window_step'])

    def process(self, df) -> pd.DataFrame:
        input_path = self.config['input_dir']
        paths_list = get_audio_files_paths(input_path, '.wav')

        df = pd.DataFrame()
        for f in paths_list:
            try:
                embedding = self.inference_emb(f)
                tmp = pd.DataFrame(np.mean(embedding, axis=0)).T
                tmp['filename'] = f.split("/")[-1]
                df = pd.concat([tmp, df], ignore_index=True)
            except RuntimeError as e:
                self.logger.error(f'An error occured in {f}: {e}')
        return df
        # df_pyannote = df
        # X_test_pyannote = df_pyannote.drop(['filename'], axis=1)
        # y_test_pyannote = df_pyannote[['filename']]
