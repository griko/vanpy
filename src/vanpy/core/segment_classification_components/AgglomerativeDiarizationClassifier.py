import time

import numpy as np
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.segment_classification_components.BaseClassificationComponent import BaseClassificationComponent
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeDiarizationClassifier(BaseClassificationComponent):
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
        payload_df_normalized = payload_df[features_columns].apply(lambda x: normalize(x.values.reshape(1, -1), norm='l2').reshape(-1) if all(x.notnull()) else np.array([-100 for _ in x]).reshape(-1), axis=1)
        payload_arr_normalized = []
        for r in payload_df_normalized.iteritems():
            payload_arr_normalized.append(r[1])
        payload_df_normalized = np.array(payload_arr_normalized)
        self.config['records_count'] = len(payload_df)

        payload_df[self.classification_column_name] = None
        if payload_df.empty:
            ComponentPayload(metadata=payload_metadata, df=payload_df)

        t_start = time.time()
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters, distance_threshold=self.threshold).fit(payload_df_normalized)
        performance_metric = [time.time() - t_start] * len(clustering.labels_)
        payload_df[self.classification_column_name] = [f'SPEAKER_{i}' if -100 not in payload_df_normalized[i] else '' for i in clustering.labels_]
        payload_metadata['classification_columns'].extend([self.classification_column_name])
        if self.config.get('performance_measurement', True):
            file_performance_column_name = f'perf_{self.get_name()}_get_diarization'
            payload_df[file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
