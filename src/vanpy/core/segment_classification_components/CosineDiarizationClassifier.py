import time
import torch
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.DisjointSet import DisjointSet


class CosineDiarizationClassifier(PipelineComponent):
    model = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='cosine_distance_diarization',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.threshold = self.config.get('threshold', 0.25)
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
        self.config['records_count'] = len(payload_df)

        payload_df[self.classification_column_name] = None
        if payload_df.empty:
            ComponentPayload(metadata=payload_metadata, df=payload_df)
        ds = DisjointSet(self.config['records_count'])
        performance_metric = []

        for i in range(self.config['records_count']):
            t_start_transcribing = time.time()
            emb1 = payload_df.iloc[i][features_columns]
            for j in range(self.config['records_count']):
                emb2 = payload_df.iloc[j][features_columns]
                if None in emb1.values or None in emb2.values:  # empty (None) row
                    continue
                if i != j and self.similarity(torch.Tensor(emb1), torch.Tensor(emb2)) > self.threshold:
                    ds.union(i, j)
                    break
            t_end_transcribing = time.time()
            performance_metric.append(t_end_transcribing - t_start_transcribing)
            self.latent_info_log(
                    f'Diarization done in {t_end_transcribing - t_start_transcribing} seconds, {i + 1}/{self.config["records_count"]}',
                    iteration=i)

        group_indexes = [f'SPEAKER_{i}' for i in ds.calculate_group_index()]
        payload_df[self.classification_column_name] = group_indexes
        payload_metadata['classification_columns'].extend([self.classification_column_name])
        if self.config.get('performance_measurement', True):
            file_performance_column_name = f'perf_{self.get_name()}_get_diarization'
            payload_df[file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
