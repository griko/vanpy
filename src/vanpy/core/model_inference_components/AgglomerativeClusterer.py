import time

import numpy as np
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.model_inference_components.BaseClassificationComponent import BaseClassificationComponent
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeClusterer(BaseClassificationComponent):
    model = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='agglomerative_clustering_diarization',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')
        self.threshold = self.config.get('threshold', 2.3)
        self.n_clusters = self.config.get('n_clusters', 3)
        self.requested_feature_list = self.build_requested_feature_list()

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
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
            file_performance_column_name = f'perf_{self.get_name()}_get_diarization'
            payload_df[file_performance_column_name] = None

        if payload_df.empty or not valid_rows_mask.any():
            return ComponentPayload(metadata=payload_metadata, df=payload_df)

        # Get indices of valid rows
        valid_indices = valid_rows_mask[valid_rows_mask].index

        # Normalize only valid rows
        valid_features = payload_df.loc[valid_indices, features_columns].values
        valid_features_normalized = normalize(valid_features, norm='l2')

        # Perform clustering only on valid rows
        t_start = time.time()
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.threshold
        ).fit(valid_features_normalized)
        processing_time = time.time() - t_start

        # Assign speaker labels only to valid rows
        speaker_labels = [f'SPEAKER_{label}' for label in clustering.labels_]
        payload_df.loc[valid_indices, self.classification_column_name] = speaker_labels

        # Add performance measurements if enabled
        if self.config.get('performance_measurement', True):
            payload_df.loc[valid_indices, file_performance_column_name] = processing_time
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        payload_metadata['classification_columns'].extend([self.classification_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
