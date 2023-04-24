import pickle
from yaml import YAMLObject

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import cached_download, create_dirs_if_not_exist


class RavdessEmotionClassifier(PipelineComponent):
    model = None
    transformer = None
    label_conversion_list = ['angry', 'disgust', 'fearful', 'happy', 'neutral/calm', 'sad', 'surprised']
    label_conversion_dict = {v: i for i, v in zip(range(len(label_conversion_list)), label_conversion_list)}
    classification_column_name: str = ''
    verbal_labels: bool = True

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='vanpy_ravdess_emotion',
                         yaml_config=yaml_config)
        self.verbal_labels = self.config.get('verbal_labels', True)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')

    def load_model(self):
        self.logger.info("Loading 7-class SVM emotion classification model, trained on RAVDESS dataset with speech_brain embedding [192 features]")
        model_path = cached_download('https://drive.google.com/uc?id=1-kQ7eschXQeYiK7wpLTrBnVv6PTSZPfO',
                                     f'{self.pretrained_models_dir}/ravdess_svm_speechbrain_ecapa_voxceleb_no_processor_cv.pkl')
        self.model = pickle.load(open(model_path, "rb"))

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata = input_payload.metadata
        payload_df = input_payload.df

        expected_columns = [f'{i}_speechbrain_embedding' for i in range(192)]  # expecting features_columns to be ['0_speechbrain_embedding','1_speechbrain_embedding',...'191_speechbrain_embedding']
        if set(expected_columns) - set(payload_df.columns):
            self.logger.error("There are no speechbrain_embedding columns in the payload, please add 'speechbrain_embedding' component to the Pipeline with 'spkrec-ecapa-voxceleb' model (or without model mentioning)")
            return input_payload
        else:
            self.logger.info("Found SpeechBrainEmbedding features in the payload, continuing with classification")

        X = payload_df[expected_columns].convert_dtypes()
        nan_idxs = X[X.isna().any(axis=1)].index
        X = X.fillna(0)
        y_pred = self.model.predict(X)
        classification_column_type = None
        if self.verbal_labels:
            payload_df[self.classification_column_name] = y_pred
        else:
            payload_df[self.classification_column_name] = [self.label_conversion_dict[x] for x in y_pred]
            classification_column_type = 'int'
        payload_df.loc[nan_idxs, self.classification_column_name] = None
        if classification_column_type:
            payload_df[self.classification_column_name] = payload_df[self.classification_column_name].astype(classification_column_type)

        payload_metadata['classification_columns'].extend([self.classification_column_name])
        return ComponentPayload(metadata=payload_metadata, df=payload_df)
