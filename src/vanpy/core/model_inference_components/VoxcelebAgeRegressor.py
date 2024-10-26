import pickle
import keras
import numpy as np
import joblib
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
        super().__init__(component_type='segment_classifier', component_name='vanpy_age',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name', f'{self.component_name}_estimation')

    def load_model(self):
        model_name = self.config.get('model', 'ann_ecapa_192_sb_librosa_31_combined')
        if model_name == 'svr_ecapa_192_sb_voxceleb':
            self.logger.info(
                "Loading SVR age regression model, trained on Voxceleb2 dataset with speech_brain embedding [192 features]")
            model_path = cached_download('https://drive.google.com/uc?id=1I6z0gjhKlzajbuKNGN5XuNiwrhxNYT8g',
                                         f'{self.pretrained_models_dir}/vc_auto_svr_reg_bal_speechbrain_ecapa_192_age_optuna.pkl')
            transformer_path = cached_download('https://drive.google.com/uc?id=1xNgnqcRJWcEhiOcWGBXEPR4pzt33xQIh',
                                               f'{self.pretrained_models_dir}/scaler_auto_vc_reg_bal_speechbrain_ecapa_192_age_optuna.pkl')
            self.model = joblib.load(model_path)

        elif model_name == 'svr_ecapa_192_sb_librosa_31_voxceleb':
            self.logger.info(
                "Loading SVR age regression model, trained on Voxceleb2 dataset with speech_brain embedding [192 features] and 31 Librosa features")
            model_path = cached_download('https://drive.google.com/uc?id=1j02hQry3lflQ-uOmdCTjWADag2BH4zmZ',
                                         f'{self.pretrained_models_dir}/vc_auto_svr_reg_bal_librosa_233_age_optuna.pkl')
            transformer_path = cached_download('https://drive.google.com/uc?id=1td-cALVUrYoKWzQU0u-dk-ygx5ClWowC',
                                               f'{self.pretrained_models_dir}/scaler_auto_vc_reg_bal_librosa_233_age_optuna.pkl')
            self.model = joblib.load(model_path)

        elif model_name == 'ann_ecapa_192_sb_timit':
            self.logger.info(
                "Loading ANN age regression model, trained on TIMIT dataset with speech_brain embedding [192 features]")
            model_path = cached_download('https://drive.google.com/uc?id=1rftvTyl223czkKXC2jfxX_rAMhm3LbUe',
                                         f'{self.pretrained_models_dir}/timit_auto_ann_reg_timit_speechbrain_ecapa_192_age_optuna.h5')
            transformer_path = cached_download('https://drive.google.com/uc?id=1kRFitAp4EryFFnEF-SbIWTMlv9inKDYg',
                                               f'{self.pretrained_models_dir}/scaler_auto_timit_reg_timit_speechbrain_ecapa_192_age_optuna.pkl')

            self.model = keras.models.load_model(model_path, compile=False)
            self.model.compile(loss='mse')
            # self.model.compile(optimizer='adam', loss='mean_absolute_error')


        elif model_name == 'ann_ecapa_192_sb_librosa_31_combined':
            self.logger.info(
                "Loading ANN age regression model, trained on combined Voxceleb2 and TIMIT datasets with speech_brain embedding [192 features] and 31 Librosa features")
            model_path = cached_download('https://drive.google.com/uc?id=1lrex6dAXsp-4AH5QfJzb2dTSqf_UF_vk',
                                         f'{self.pretrained_models_dir}/combined_ann_reg_librosa_233_age_optuna.h5')
            transformer_path = cached_download('https://drive.google.com/uc?id=1XcR83B8WAtfI5wClCKxGLatezCumLYnH',
                                               f'{self.pretrained_models_dir}/scaler_combined_reg_librosa_233_age_optuna.pkl')

            self.model = keras.models.load_model(model_path, compile=False)
            self.model.compile(loss='mse')

        else:
            raise ValueError(f"Unknown model name: {model_name}, choose from 'svr_ecapa_192_sb_voxceleb', 'svr_ecapa_192_sb_librosa_31_voxceleb', 'ann_ecapa_192_sb_timit', 'ann_ecapa_192_sb_librosa_31_combined'")



        # self.logger.info("Loading ANN age regression model, trained on Voxceleb2 dataset with speech_brain embedding [192 features]")
        # model_path = cached_download('https://drive.google.com/uc?id=1L8RcC788KK6dJi4YmLO_yjQA06nS37rO',
        #                              f'{self.pretrained_models_dir}/ann_age_speechbrain_ecapa_voxceleb_processor_0.76.5.h5')
        # self.model = keras.models.load_model(model_path, compile=False)
        # self.model.compile(optimizer='adam', loss='mean_absolute_error')
        # transformer_path = cached_download('https://drive.google.com/uc?id=153HsLcWaQbR-aMVRFS7TVut8julZkJhY',
        #                              f'{self.pretrained_models_dir}/processor_age_speechbrain_ecapa_voxceleb_processor_0.76.5.pkl')
        self.transformer = pickle.load(open(transformer_path, "rb"))

        self.expected_feature_columns = [f'{i}_speechbrain_embedding' for i in range(192)]
        if model_name in ['svr_ecapa_192_sb_librosa_31_voxceleb', 'ann_ecapa_192_sb_librosa_31_combined']:
            librosa_columns = ['zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
                               'spectral_flatness', 'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
                               'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'd_mfcc_0',
                               'd_mfcc_1', 'd_mfcc_2', 'd_mfcc_3', 'd_mfcc_4', 'd_mfcc_5', 'd_mfcc_6', 'd_mfcc_7',
                               'd_mfcc_8', 'd_mfcc_9', 'd_mfcc_10', 'd_mfcc_11', 'd_mfcc_12']
            self.expected_feature_columns.extend(librosa_columns)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata = input_payload.metadata
        payload_df = input_payload.df

        if set(self.expected_feature_columns) - set(payload_df.columns):
            self.logger.error("There are not enough features in the payload")
            return input_payload
        else:
            self.logger.info("Found required features in the payload, continuing with classification")

        X = payload_df[self.expected_feature_columns].convert_dtypes()
        nan_idxs = X[X.isna().any(axis=1)].index
        X = X.fillna(0)

        if self.config.get('apply_transform', False):
            X = self.transformer.transform(X)
        else:
            self.logger.info(
                "Skipping transformation of features, FIY: the model was trained on transformed features with a StandardScaler, make sure to apply the same transformation to the input features")

        # X.columns = X.columns.astype(str)  # expecting features_columns to be ['0_speechbrain_embedding','1_speechbrain_embedding',...'191_speechbrain_embedding']
        # y_pred = self.model.predict(np.array(X).astype("float32"))
        y_pred = self.model.predict(X)
        payload_df[self.classification_column_name] = y_pred.reshape(-1)
        payload_df.loc[nan_idxs, self.classification_column_name] = None
        payload_metadata['classification_columns'].extend([self.classification_column_name])
        return ComponentPayload(metadata=payload_metadata, df=payload_df)
