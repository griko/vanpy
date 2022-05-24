import pickle
from typing import Tuple
from yaml import YAMLObject
from core.PiplineComponent import PipelineComponent, ComponentPayload
from pyannote.audio import Inference
import numpy as np
import pandas as pd

from utils.utils import get_audio_files_paths, cached_download


class CVGenderClassifier(PipelineComponent):
    is_model_loaded = False
    simple_imputer = None
    xgb_cl = None
    transformer = None
    label_conversion_dict = {0: 'female', 1: 'male'}
    payload_column_name = 'common_voices_gender_classification'

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='speaker_classifier', component_name='common_voices_gender',
                         yaml_config=yaml_config)
        self.inference_emb = Inference("pyannote%2Fembedding",
                                  window="sliding",
                                  duration=self.config['sliding_window_duration'], step=self.config['sliding_window_step'])

    def load_model(self):
        self.logger.info("Loading XGBoost gender classification model, trained on Mozilla Common Voice dataset with pyannote2.0 embedding [512 features]")
        if not self.is_model_loaded:
            imputer_path = cached_download('https://drive.google.com/uc?id=1X_CiiF5rcvl1n9S9EHrDJGjXUDgWmGiz',
                                           'pretrained_models/common_voice/xgb_gender_512_simple_imputer.pkl')
            model_path = cached_download('https://drive.google.com/uc?id=1AVxCtZqccWhCJ4uVJb13WViuWS9e9uBi',
                                         'pretrained_models/common_voice/xgb_gender_512_model.pkl')
            transformer_path = cached_download('https://drive.google.com/uc?id=1bCiT5YGQd_CmpvA5DjoDUGZRl53TkbwE',
                                               'pretrained_models/common_voice/xgb_gender_512_full_processor.pkl')
            self.simple_imputer = pickle.load(open(imputer_path, "rb"))
            self.xgb_cl = pickle.load(open(model_path, "rb"))
            self.transformer = pickle.load(open(transformer_path, "rb"))

    def process(self, input_object: ComponentPayload) -> ComponentPayload:
        self.load_model()
        payload_features = input_object.features
        payload_df = input_object.df
        X = payload_df[payload_features['feature_columns']]
        X.columns = X.columns.astype(str)  # expecting features_columns to be ['0','1',...'511']
        X = self.transformer.transform(X)
        # y_test = simple_imputer.transform(
        #     y_test_pyannote.values.reshape(-1, 1)
        # )
        y_pred = self.xgb_cl.predict(X)
        payload_df[self.payload_column_name] = [self.label_conversion_dict[x] for x in y_pred]
        payload_features['classification_columns'].extend([self.payload_column_name])
        return ComponentPayload(features=payload_features, df=payload_df)
