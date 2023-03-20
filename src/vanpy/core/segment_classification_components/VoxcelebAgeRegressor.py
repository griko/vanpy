import pickle
import keras
from yaml import YAMLObject

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import cached_download, create_dirs_if_not_exist


class VoxcelebAgeRegressor(PipelineComponent):
    model = None
    transformer = None
    classification_column_name: str = ''
    verbal_labels: bool = False

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='vanpy_voxceleb_age',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config['classification_column_name']
        self.pretrained_models_dir = self.config['pretrained_models_dir']

    def load_model(self):
        self.logger.info("Loading ANN age regression model, trained on Voxceleb2 dataset with speech_brain embedding [192 features]")
        model_path = cached_download('https://drive.google.com/uc?id=1L8RcC788KK6dJi4YmLO_yjQA06nS37rO',
                                     f'{self.pretrained_models_dir}/ann_age_speechbrain_ecapa_voxceleb_processor_0.76.5.h5')
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='mean_absolute_error')
        transformer_path = cached_download('https://drive.google.com/uc?id=153HsLcWaQbR-aMVRFS7TVut8julZkJhY',
                                     f'{self.pretrained_models_dir}/processor_age_speechbrain_ecapa_voxceleb_processor_0.76.5.pkl')
        self.transformer = pickle.load(open(transformer_path, "rb"))

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata = input_payload.metadata
        payload_df = input_payload.df

        expected_columns = [f'{i}_speechbrain_embedding' for i in range(192)]
        if set(expected_columns) - set(payload_df.columns):
            self.logger.error("There are no speechbrain_embedding columns in the payload, please add 'speechbrain_embedding' component to the Pipeline with 'spkrec-ecapa-voxceleb' model (or without model mentioning)")
            return input_payload
        else:
            self.logger.info("Found SpeechBrainEmbedding features in the payload, continuing with classification")

        # X = payload_df[payload_metadata['feature_columns']]
        X = payload_df[expected_columns].convert_dtypes()
        X = self.transformer.transform(X)
        # X.columns = X.columns.astype(str)  # expecting features_columns to be ['0_speechbrain_embedding','1_speechbrain_embedding',...'191_speechbrain_embedding']
        y_pred = self.model.predict(X)
        payload_df[self.classification_column_name] = y_pred.reshape(-1)
        payload_metadata['classification_columns'].extend([self.classification_column_name])
        return ComponentPayload(metadata=payload_metadata, df=payload_df)
