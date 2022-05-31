import pickle
from yaml import YAMLObject
from core.PiplineComponent import PipelineComponent, ComponentPayload
from utils.utils import cached_download


class CVGenderClassifier(PipelineComponent):
    model = None
    transformer = None
    label_conversion_dict = {0: 'female', 1: 'male'}
    classification_column_name: str = ''
    verbal_labels: bool = False

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='common_voices_gender',
                         yaml_config=yaml_config)
        self.verbal_labels = self.config['verbal_labels']
        self.classification_column_name = self.config['classification_column_name']
        self.pretrained_models_dir = self.config['pretrained_models_dir']

    def load_model(self):
        self.logger.info("Loading XGBoost gender classification model, trained on Mozilla Common Voice "
                         "dataset with pyannote2.0 embedding [512 features]")
        model_path = cached_download('https://drive.google.com/uc?id=1AVxCtZqccWhCJ4uVJb13WViuWS9e9uBi',
                                      f'{self.pretrained_models_dir}/xgb_gender_512_model.pkl')
        transformer_path = cached_download('https://drive.google.com/uc?id=1bCiT5YGQd_CmpvA5DjoDUGZRl53TkbwE',
                                            f'{self.pretrained_models_dir}/xgb_gender_512_full_processor.pkl')
        self.model = pickle.load(open(model_path, "rb"))
        self.transformer = pickle.load(open(transformer_path, "rb"))

    def process(self, input_object: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_features = input_object.features
        payload_df = input_object.df
        X = payload_df[payload_features['feature_columns']]
        X.columns = X.columns.astype(str)  # expecting features_columns to be ['0','1',...'511']
        X = self.transformer.transform(X)
        y_pred = self.model.predict(X)
        payload_df[self.classification_column_name] = \
            [self.label_conversion_dict[x] if self.verbal_labels else x for x in y_pred]
        payload_features['classification_columns'].extend([self.classification_column_name])
        return ComponentPayload(features=payload_features, df=payload_df)
