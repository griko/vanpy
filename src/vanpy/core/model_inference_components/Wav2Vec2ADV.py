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

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='wav2vec2adv',
                         yaml_config=yaml_config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampling_rate = self.config.get('sampling_rate', 16000)

    def load_model(self):
        import torch
        self.logger.info("Loading wav2vec 2.0 arousal, dominance and valence prediction model")
        self.processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", cache_dir=self.pretrained_models_dir)
        self.model = EmotionModel.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", cache_dir=self.pretrained_models_dir)
        self.model.to(self.device)
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

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

    def process_item(self, f, input_column):
        try:
            # Loading the audio file
            audio, rate = librosa.load(f, sr=self.sampling_rate)
            arousal, dominance, valence = self.process_func(np.reshape(audio, [1, len(audio)]), rate)[0]

            # Create a DataFrame to hold the results
            f_df = pd.DataFrame({
                input_column: [f],
                'arousal': [arousal],
                'dominance': [dominance],
                'valence': [valence],
            })

        except (FileNotFoundError, RuntimeError, TypeError, EOFError) as e:
            self.logger.error(f"An error occurred in {f}: {e}")
            f_df = pd.DataFrame({
                input_column: [f],
                'arousal': [None],
                'dominance': [None],
                'valence': [None],
            })

        return f_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process the input payload and return the output payload.
        :param input_payload: The input payload.
        :return: The output payload.
        """
        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].dropna().tolist()
        records_count = len(paths_list)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload

        # Define which columns should be in the metadata
        payload_metadata = self.add_performance_column_to_metadata(payload_metadata)
        payload_metadata = self.add_classification_columns_to_metadata(payload_metadata, ['arousal', 'dominance', 'valence'])

        # Call process_with_progress
        p_df = self.process_with_progress(paths_list, payload_metadata, input_column)

        # Merge the processed DataFrame back into the original DataFrame
        payload_df = pd.merge(left=payload_df, right=p_df, how='left', left_on=input_column, right_on=input_column)

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
