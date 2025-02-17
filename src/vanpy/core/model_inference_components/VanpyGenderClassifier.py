import pickle
from yaml import YAMLObject

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import cached_download, create_dirs_if_not_exist


class VanpyGenderClassifier(PipelineComponent):
    """
    Binary gender classification component using SVM models trained on speech embeddings.

    :ivar model: Loaded classification model instance.
    :ivar transformer: Feature transformation pipeline instance.
    :ivar label_conversion_list: List of string labels for gender classes.
    :ivar label_conversion_dict: Dictionary mapping numeric indices to gender labels.
    :ivar classification_column_name: Name of the output classification column.
    :ivar verbal_labels: Whether to use string labels (True) or numeric indices (False).
    """
    model = None
    transformer = None
    label_conversion_list = ['female', 'male']
    label_conversion_dict = {i: v for i, v in zip(range(len(label_conversion_list)), label_conversion_list)}
    classification_column_name: str = ''
    verbal_labels: bool = True

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the gender classifier component.

        :param yaml_config: Configuration parameters for the classifier.
        """
        super().__init__(component_type='segment_classifier', component_name='vanpy_gender',
                         yaml_config=yaml_config)
        self.verbal_labels = self.config.get('verbal_labels', True)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')

    def load_model(self):
        """
        Load the gender classification model and its feature transformer.
        
        :raises ValueError: If an unknown model name is provided.
        """
        model_name = self.config.get('model', 'svm_ecapa_192_sb_voxceleb')
        if model_name == 'svm_ecapa_192_sb_voxceleb':

            self.logger.info("Loading SVM gender classification model, trained on Voxceleb2 dataset with ECAPA-TDNN speech_brain embedding [192 features]")
            model_path = cached_download('https://drive.google.com/uc?id=1ytf7wV1z-oarvjlAgZ1NbVE1VgsKRYBX',
                                         f'{self.pretrained_models_dir}/vc_svm_cls_bal_speechbrain_ecapa_192_gender_optuna.pkl')
            processor_path = cached_download('https://drive.google.com/uc?id=1OEeI0nqECXSiA7B_8otCW0lVYF0bFWmW',
                                            f'{self.pretrained_models_dir}/processor_vc_svm_cls_bal_speechbrain_ecapa_192_gender_optuna.pkl')
            self.expected_feature_columns = [f'{i}_speechbrain_embedding' for i in range(192)]  # expecting features_columns to be ['0_speechbrain_embedding','1_speechbrain_embedding',...'191_speechbrain_embedding']
        elif model_name == 'svm_xvect_512_sb_voxceleb':
            self.logger.info(
                "Loading SVM gender classification model, trained on Voxceleb2 dataset with XVECT speech_brain embedding [512 features]")
            model_path = cached_download('https://drive.google.com/uc?id=1YkGDW-PZSPkuMqX0GNEhNwWKoMjq9C7X',
                                         f'{self.pretrained_models_dir}/vc_svm_cls_bal_speechbrain_xvect_512_gender_optuna.pkl')
            processor_path = cached_download('https://drive.google.com/uc?id=1pww7on1En7sU26-3oWl3HfGfg426g7hX',
                                             f'{self.pretrained_models_dir}/processor_vc_svm_cls_bal_speechbrain_xvect_512_gender_optuna.pkl')
            self.expected_feature_columns = [f'{i}_speechbrain_embedding' for i in range(
                512)]  # expecting features_columns to be ['0_speechbrain_embedding','1_speechbrain_embedding',...'191_speechbrain_embedding']
        else:
            raise ValueError(f"Unknown model name: {model_name}, choose from 'svm_ecapa_192_sb_voxceleb', 'svm_xvect_512_sb_voxceleb'")

        self.model = pickle.load(open(model_path, "rb"))
        self.transformer = pickle.load(open(processor_path, "rb"))

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process audio features to predict gender classifications.

        :param input_payload: Input payload containing audio features and metadata.
        :return: Output payload containing gender classifications.
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
            self.logger.info("Skipping transformation of features, FIY: the model was trained on transformed features with a StandardScaler, make sure to apply the same transformation to the input features")

        y_pred = self.model.predict(X)
        if self.verbal_labels:
            payload_df[self.classification_column_name] = [self.label_conversion_dict[x] for x in y_pred]
        else:
            payload_df[self.classification_column_name] = y_pred
        payload_df.loc[nan_idxs, self.classification_column_name] = None

        payload_metadata['classification_columns'].extend([self.classification_column_name])
        return ComponentPayload(metadata=payload_metadata, df=payload_df)
