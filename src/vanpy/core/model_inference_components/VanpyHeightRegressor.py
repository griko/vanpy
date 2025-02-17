import pickle

import joblib
import keras
from yaml import YAMLObject

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import cached_download, create_dirs_if_not_exist


class VanpybHeightRegressor(PipelineComponent):
    """
    Height regression component using SVR model trained on speech embeddings.

    :ivar model: Loaded regression model instance.
    :ivar transformer: Feature transformation pipeline instance.
    :ivar classification_column_name: Name of the output estimation column.
    """
    model = None
    transformer = None
    classification_column_name: str = ''
    verbal_labels: bool = False

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the height regressor component.

        :param yaml_config: Configuration parameters for the regressor.
        """
        super().__init__(component_type='segment_classifier', component_name='vanpy_height',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name', f'{self.component_name}_estimation')

    def load_model(self):
        """
        Load the height regression model and its feature transformer.
        """
        self.logger.info("Loading SVR height regression model, trained on Voxceleb2 dataset with speech_brain embedding [192 features]")
        model_path = cached_download('https://drive.google.com/uc?id=1WWK4h-wTlCpuIQi2antiPEg3r8n4Hxot',
                                     f'{self.pretrained_models_dir}/vc_auto_svr_reg_bal_speechbrain_ecapa_192_height_optuna.pkl')
        self.model = joblib.load(model_path)
        transformer_path = cached_download('https://drive.google.com/uc?id=1OPx4gXPpDhZ_8QSv9m5LWAo1yfr_H4aL',
                                     f'{self.pretrained_models_dir}/scaler_auto_vc_reg_bal_speechbrain_ecapa_192_height_optuna.pkl')
        self.transformer = pickle.load(open(transformer_path, "rb"))
        self.expected_feature_columns = [f'{i}_speechbrain_embedding' for i in range(192)]

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process audio features to predict height estimations.

        :param input_payload: Input payload containing audio features and metadata.
        :return: Output payload containing height estimations.
        """
        if not self.model:
            self.load_model()

        payload_metadata = input_payload.metadata
        payload_df = input_payload.df


        if set(self.expected_feature_columns) - set(payload_df.columns):
            self.logger.error("There are no speechbrain_embedding columns in the payload, please add 'speechbrain_embedding' component to the Pipeline with 'spkrec-ecapa-voxceleb' model (or without model mentioning)")
            return input_payload
        else:
            self.logger.info("Found SpeechBrainEmbedding features in the payload, continuing with classification")

        X = payload_df[self.expected_feature_columns].convert_dtypes()
        nan_idxs = X[X.isna().any(axis=1)].index
        X = X.fillna(0)

        if self.config.get('apply_transform', False):
            X = self.transformer.transform(X)
        else:
            self.logger.info(
                "Skipping transformation of features, FIY: the model was trained on transformed features with a StandardScaler, make sure to apply the same transformation to the input features")

        # X.columns = X.columns.astype(str)  # expecting features_columns to be ['0_speechbrain_embedding','1_speechbrain_embedding',...'191_speechbrain_embedding']
        y_pred = self.model.predict(X)
        payload_df[self.classification_column_name] = y_pred.reshape(-1)
        payload_df.loc[nan_idxs, self.classification_column_name] = None
        payload_metadata['classification_columns'].extend([self.classification_column_name])
        return ComponentPayload(metadata=payload_metadata, df=payload_df)
