from sklearn.mixture import GaussianMixture
from vanpy.core.ComponentPayload import ComponentPayload
from yaml import YAMLObject
import time
from vanpy.core.segment_classification_components.BaseClassificationComponent import BaseClassificationComponent


class GMMDiarizationClassifier(BaseClassificationComponent):
    model = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='gmm_clustering_diarization',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')
        self.n_components = self.config.get('n_components', 3)  # Number of Gaussian components
        self.covariance_type = self.config.get('covariance_type', 'full')  # Covariance type
        self.requested_feature_list = self.build_requested_feature_list()

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        payload_metadata, payload_df = input_payload.unpack()
        self.config['records_count'] = len(payload_df)

        payload_df[self.classification_column_name] = None
        if payload_df.empty:
            return ComponentPayload(metadata=payload_metadata, df=payload_df)

        performance_metric = []
        X = payload_df[self.requested_feature_list].fillna(-100).values
        gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type)
        gmm.fit(X)

        for i in range(self.config['records_count']):
            t_start_transcribing = time.time()
            emb = X[i, :]
            label = f'SPEAKER_{gmm.predict(emb.reshape(1, -1))[0]}' if -100 not in emb else ''
            payload_df.at[i, self.classification_column_name] = label
            t_end_transcribing = time.time()
            performance_metric.append(t_end_transcribing - t_start_transcribing)
            self.latent_info_log(
                f'GMM clustering done in {t_end_transcribing - t_start_transcribing} seconds, {i + 1}/{self.config["records_count"]}',
                iteration=i)

        payload_metadata['classification_columns'].extend([self.classification_column_name])
        if self.config.get('performance_measurement', True):
            file_performance_column_name = f'perf_{self.get_name()}_get_cluster'
            payload_df[file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)