import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from yaml import YAMLObject
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

class Wav2Vec2ADV(PipelineComponent):
    # A prediction model for arousal, dominance and valence
    model = None
    tokenizer = None
    pretrained_models_dir: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='wav2vec2adv',
                         yaml_config=yaml_config)
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.pretrained_models_dir = self.config['pretrained_models_dir']

    def load_model(self):
        self.logger.info("Loading wav2vec 2.0 arousal, dominance and valence prediction model")
        self.processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", cache_dir=self.pretrained_models_dir)
        self.model = EmotionModel.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", cache_dir=self.pretrained_models_dir)

    def process_func(self,
        x: np.ndarray,
        sampling_rate: int,
    ) -> np.ndarray:
        r"""Predict emotions or extract arousal, dominance and valence from raw audio signal."""

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = self.processor(x, sampling_rate=sampling_rate)
        y = y['input_values'][0]
        y = torch.from_numpy(y).to(self.device)

        # run through model
        with torch.no_grad():
            y = self.model(y)[1]

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].tolist()
        self.config['records_count'] = len(paths_list)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload
        
        prediction = []
        performance_metric = []
        sr = self.config['sampling_rate'] if 'sampling_rate' in self.config else 16000
        for j, f in enumerate(paths_list):
            try:
                t_start_transcribing = time.time()

                # Loading the audio file
                audio, rate = librosa.load(f, sr=sr)
                arousal, dominance, valence = self.process_func(np.reshape(audio, [1, len(audio)]), rate)[0]
                prediction.append((arousal, dominance, valence))
                t_end_transcribing = time.time()
                performance_metric.append(t_end_transcribing - t_start_transcribing)
                self.latent_info_log(
                    f'Transcribed {f} in {t_end_transcribing - t_start_transcribing} seconds, {j + 1}/{len(paths_list)}',
                    iteration=j)
            except RuntimeError as e:
                prediction.append((None, None, None))
                performance_metric.append(float('inf'))
                self.logger.error(f'An error occurred in {f}, {j + 1}/{len(paths_list)}: {e}')

        columns = ['arousal', 'dominance', 'valence']
        prediction_df = pd.DataFrame(prediction, columns=columns)
        for col in prediction_df.columns:
            payload_df[col] = prediction_df[col]
        payload_metadata['classification_columns'].extend([columns])
        if 'performance_measurement' in self. config and self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_transcription'
            payload_df[file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
