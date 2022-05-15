import pickle
from typing import Tuple
from yaml import YAMLObject
from core.PiplineComponent import PipelineComponent
from pyannote.audio import Inference
import numpy as np
import pandas as pd

from utils.utils import get_audio_files_paths, cached_download


class CVGenderClassifier(PipelineComponent):

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='speaker_classifier', component_name='common_voices_gender',
                         yaml_config=yaml_config)
        self.inference_emb = Inference("pyannote%2Fembedding",
                                  window="sliding",
                                  duration=self.config['sliding_window_duration'], step=self.config['sliding_window_step'])
        imputer_path = cached_download('https://drive.google.com/file/d/1X_CiiF5rcvl1n9S9EHrDJGjXUDgWmGiz/view?usp=sharing', 'pretrained_models/common_voice/xgb_gender_512_simple_imputer.pkl')
        model_path = cached_download('https://drive.google.com/file/d/1AVxCtZqccWhCJ4uVJb13WViuWS9e9uBi/view?usp=sharing', 'pretrained_models/common_voice/xgb_gender_512_model.pkl')
        transformer_path = cached_download('https://drive.google.com/file/d/1bCiT5YGQd_CmpvA5DjoDUGZRl53TkbwE/view?usp=sharing', 'pretrained_models/common_voice/xgb_gender_512_full_processor.pkl')
        self.simple_imputer = pickle.load(open(imputer_path, "rb"))
        self.xgb_cl = pickle.load(open(model_path, "rb"))
        self.transformer = pickle.load(open(transformer_path, "rb"))


    def process(self, input_dir: str = '', df: pd.DataFrame = None) -> Tuple[str, pd.DataFrame]:

        X_test_pyannote.columns = X_test_pyannote.columns.astype(str)
        X_test = full_processor.transform(X_test_pyannote)
        y_test = simple_imputer.transform(
            y_test_pyannote.values.reshape(-1, 1)
        )
        y_pred = xgb_cl.predict(X_test)
        y_pred


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
