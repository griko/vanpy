import time
import torch
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.DisjointSet import DisjointSet
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeDiarizationClassifier(PipelineComponent):
    model = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='agglomerative_clustering_diarization',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')
        self.threshold = self.config.get('threshold', 2.3)
        self.requested_feature_list = self.build_requested_feature_list()

    def build_requested_feature_list(self):
        features_list = []
        if 'features_list' in self.config:
            for feature in self.config['features_list']:
                if isinstance(feature, str):
                    features_list.append(feature)
                elif isinstance(feature, dict):
                    key = tuple(feature.keys())[0]
                    if 'start_index' not in feature[key] or 'stop_index' not in feature[key]:
                        raise AttributeError('Invalid form of multiple-index feature. You have to supply start_index and stop_index')
                    for i in range(int(feature[key]['start_index']), int(feature[key]['stop_index'])):
                        features_list.append(f'{i}_{key}')
        return features_list

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        payload_metadata, payload_df = input_payload.unpack()
        features_columns = [column for column in payload_df.columns if column in self.requested_feature_list]
        payload_df_normalized = normalize(payload_df[features_columns], axis=1, norm='l2')
        self.config['records_count'] = len(payload_df)

        payload_df[self.classification_column_name] = None
        if payload_df.empty:
            ComponentPayload(metadata=payload_metadata, df=payload_df)

        t_start = time.time()
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.threshold).fit(payload_df_normalized)
        performance_metric = [time.time() - t_start] * len(clustering.labels_)
        payload_df[self.classification_column_name] = [f'SPEAKER_{i}' for i in clustering.labels_]
        payload_metadata['classification_columns'].extend([self.classification_column_name])
        if self.config.get('performance_measurement', True):
            file_performance_column_name = f'perf_{self.get_name()}_get_diarization'
            payload_df[file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
