from sklearn.mixture import GaussianMixture
from vanpy.core.ComponentPayload import ComponentPayload
from yaml import YAMLObject
import time
from vanpy.core.model_inference_components.BaseClassificationComponent import BaseClassificationComponent


class GMMClusterer(BaseClassificationComponent):
    """
    Gaussian Mixture Model clustering component for speaker diarization.
    
    :ivar model: GMM clustering model instance.
    :ivar classification_column_name: Name of the column containing speaker labels.
    :ivar n_components: Number of Gaussian components in the mixture.
    :ivar covariance_type: Type of covariance matrix to use.
    """
    model = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the GMM clusterer.

        :param yaml_config: Configuration parameters for the clusterer.
        """
        super().__init__(component_type='segment_classifier', component_name='gmm_clustering_diarization',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')
        self.n_components = self.config.get('n_components', 3)  # Number of Gaussian components
        self.covariance_type = self.config.get('covariance_type', 'full')  # Covariance type
        self.requested_feature_list = self.build_requested_feature_list()

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process input features using GMM clustering for speaker diarization.

        :param input_payload: Input payload containing features and metadata.
        :return: Output payload containing speaker diarization results.
        """
        payload_metadata, payload_df = input_payload.unpack()
        features_columns = [column for column in payload_df.columns if column in self.requested_feature_list]
        if len(features_columns) != len(self.requested_feature_list):
            self.logger.warning('Some requested features are not present in the input payload. Skipping diarization.')
            return ComponentPayload(metadata=payload_metadata, df=payload_df)

        # Create mask for rows with all features present
        valid_rows_mask = payload_df[features_columns].notna().all(axis=1)

        # Initialize columns with None
        payload_df[self.classification_column_name] = None
        if self.config.get('performance_measurement', True):
            file_performance_column_name = f'perf_{self.get_name()}_get_cluster'
            payload_df[file_performance_column_name] = None

        if payload_df.empty or not valid_rows_mask.any():
            return ComponentPayload(metadata=payload_metadata, df=payload_df)

        # Get indices and features of valid rows
        valid_indices = valid_rows_mask[valid_rows_mask].index
        valid_features = payload_df.loc[valid_indices, features_columns].values

        # Fit GMM on valid features
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type
        )
        gmm.fit(valid_features)

        # Predict all valid samples at once
        t_start = time.time()
        speaker_labels = gmm.predict(valid_features)
        processing_time = time.time() - t_start

        # Convert numerical labels to speaker strings
        speaker_strings = [f'SPEAKER_{label}' for label in speaker_labels]

        # Assign speaker labels only to valid rows
        payload_df.loc[valid_indices, self.classification_column_name] = speaker_strings

        # Add performance measurements if enabled
        if self.config.get('performance_measurement', True):
            processing_times = [processing_time / len(valid_indices)] * len(valid_indices)
            payload_df.loc[valid_indices, file_performance_column_name] = processing_times
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        payload_metadata['classification_columns'].extend([self.classification_column_name])

        if self.latent_logger_enabled:
            self.logger.info(
                f'GMM clustering completed in {processing_time} seconds for {len(valid_indices)} samples'
            )

        return ComponentPayload(metadata=payload_metadata, df=payload_df)