import time
import whisper
import torch
import librosa
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import create_dirs_if_not_exist
from vanpy.utils.DisjointSet import DisjointSet


class CosineDiarizationClassifier(PipelineComponent):
    model = None
    classification_column_name: str = ''
    pretrained_models_dir: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='cosine_distance_diarization',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config['classification_column_name'] \
            if 'classification_column_name' in self.config else 'diarization_classification'
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.threshold = self.config['threshold'] if 'threshold' in self.config else 0.25
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

        # speakers = ['SPEAKER_0']
        # speaker_idx = 0
        payload_df[self.classification_column_name] = None
        if payload_df.empty:
            ComponentPayload(metadata=payload_metadata, df=payload_df)
        ds = DisjointSet(self.config['records_count'])
        # payload_df.iloc[0][self.classification_column_name] = 'SPEAKER_0'
        performance_metric = []

        for i in range(self.config['records_count']):
            emb1 = payload_df.iloc[i][features_columns]
            for j in range(self.config['records_count']):
                emb2 = payload_df.iloc[j][features_columns]
                if i != j and self.similarity(torch.Tensor(emb1), torch.Tensor(emb2)) > self.threshold:
                    ds.union(i, j)
                    break
        group_indexes = [f'SPEAKER_{i}' for i in ds.calculate_group_index()]
        payload_df[self.classification_column_name] = group_indexes
        # for i in range(self.config['records_count']):
        #     payload_df.iloc[i][self.classification_column_name] =
            # t_start_transcribing = time.time()
            # Loading the audio file
            # audio, rate = librosa.load(f, sr=sr)
            # audio = whisper.load_audio(f, sr=sr)
            # audio = whisper.pad_or_trim(audio)
            #
            # mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            #
            # _, probs = self.model.detect_language(mel)

            # options = whisper.DecodingOptions(fp16=False)
            # transcription = whisper.decode(self.model, mel, options)
            # stts.append(transcription.text)
            # if 'detect_language' in self.config and self.config['detect_language']:
            #     languages.append(transcription.language)
            # t_end_transcribing = time.time()
            # performance_metric.append(t_end_transcribing - t_start_transcribing)
            # self.latent_info_log(
            #     f'Transcribed {f} in {t_end_transcribing - t_start_transcribing} seconds, {j + 1}/{len(paths_list)}',
            #     iteration=j)
        #
        # payload_df[self.stt_column_name] = stts
        # payload_metadata['classification_columns'].extend([self.stt_column_name])
        # if 'detect_language' in self.config and self.config['detect_language']:
        #     payload_df[self.language_classification_column_name] = languages
        #     payload_metadata['classification_columns'].extend([self.language_classification_column_name])
        # if self.config['performance_measurement']:
        #     file_performance_column_name = f'perf_{self.get_name()}_get_transcription'
        #     payload_df[file_performance_column_name] = performance_metric
        #     payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
