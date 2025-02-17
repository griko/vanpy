import time
import torch
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.model_inference_components.BaseClassificationComponent import BaseClassificationComponent
from vanpy.utils.DisjointSet import DisjointSet
from sklearn.preprocessing import normalize
import numpy as np


class CosineDistanceClusterer(BaseClassificationComponent):
    """
    A clustering component that uses cosine similarity for speaker diarization.
    
    :ivar model: Clustering model instance.
    :ivar classification_column_name: Name of the column containing speaker labels.
    :ivar similarity: Cosine similarity computation instance.
    :ivar threshold: Similarity threshold for clustering speakers.
    """
    model = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the cosine distance clusterer.

        :param yaml_config: Configuration parameters for the clusterer.
        """
        super().__init__(component_type='segment_classifier', component_name='cosine_distance_diarization',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.threshold = self.config.get('threshold', 0.25)
        self.requested_feature_list = self.build_requested_feature_list()

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process input features to cluster speakers based on cosine similarity.

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
            file_performance_column_name = f'perf_{self.get_name()}_get_diarization'
            payload_df[file_performance_column_name] = None

        if payload_df.empty or not valid_rows_mask.any():
            return ComponentPayload(metadata=payload_metadata, df=payload_df)

        # Get indices of valid rows
        valid_indices = valid_rows_mask[valid_rows_mask].index
        records_count = len(valid_indices)

        # Normalize only valid rows
        payload_df_normalized = payload_df.loc[valid_indices, features_columns].apply(
            lambda x: normalize(x.values.reshape(1, -1), norm='l2').reshape(-1),
            axis=1
        )

        ds = DisjointSet(records_count)
        performance_metric = []

        # Process only valid rows
        for i, idx_i in enumerate(valid_indices):
            t_start_transcribing = time.time()
            emb1 = payload_df_normalized.iloc[i]

            for j, idx_j in enumerate(valid_indices):
                if i == j:
                    continue

                emb2 = payload_df_normalized.iloc[j]
                if self.similarity(torch.Tensor(emb1), torch.Tensor(emb2)) > self.threshold:
                    ds.union(i, j)
                    break

            t_end_transcribing = time.time()
            performance_metric.append(t_end_transcribing - t_start_transcribing)

            self.latent_info_log(
                f'Diarization done in {t_end_transcribing - t_start_transcribing} seconds, {i + 1}/{records_count}',
                iteration=i)

        # Get group labels for valid rows
        group_indexes = [f'SPEAKER_{i}' for i in ds.calculate_group_index()]

        # Assign results only to valid rows
        payload_df.loc[valid_indices, self.classification_column_name] = group_indexes
        if self.config.get('performance_measurement', True):
            payload_df.loc[valid_indices, file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        payload_metadata['classification_columns'].extend([self.classification_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
