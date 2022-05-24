import pickle
from typing import Tuple
from yaml import YAMLObject
from core.PiplineComponent import PipelineComponent, ComponentPayload
from pyannote.audio import Inference
import numpy as np
import pandas as pd

from utils.utils import get_audio_files_paths, cached_download


class CVGenderClassifier(PipelineComponent):
    xgb_cl = None
    transformer = None
    label_conversion_dict = {0: 'female', 1: 'male'}
    classification_column_name: str = ''
    verbal_labels: bool = False

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='common_voices_gender',
                         yaml_config=yaml_config)
        self.verbal_labels = self.config['verbal_labels']
        self.classification_column_name = self.config['classification_column_name']

    def load_model(self):
        self.logger.info("Loading XGBoost gender classification model, trained on Mozilla Common Voice dataset with pyannote2.0 embedding [512 features]")
        model_path = cached_download('https://drive.google.com/uc?id=1AVxCtZqccWhCJ4uVJb13WViuWS9e9uBi',
                                     'pretrained_models/common_voice/xgb_gender_512_model.pkl')
        transformer_path = cached_download('https://drive.google.com/uc?id=1bCiT5YGQd_CmpvA5DjoDUGZRl53TkbwE',
                                           'pretrained_models/common_voice/xgb_gender_512_full_processor.pkl')
        self.xgb_cl = pickle.load(open(model_path, "rb"))
        self.transformer = pickle.load(open(transformer_path, "rb"))

    def process(self, input_object: ComponentPayload) -> ComponentPayload:
        self.load_model()
        payload_features = input_object.features
        payload_df = input_object.df
        X = payload_df[payload_features['feature_columns']]
        X.columns = X.columns.astype(str)  # expecting features_columns to be ['0','1',...'511']
        X = self.transformer.transform(X)
        y_pred = self.xgb_cl.predict(X)
        payload_df[self.classification_column_name] = [self.label_conversion_dict[x] if self.verbal_labels else x for x in y_pred]
        payload_features['classification_columns'].extend([self.classification_column_name])
        return ComponentPayload(features=payload_features, df=payload_df)
