import os
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
    """
    Neural network head for regression tasks on wav2vec features.

    :ivar dense: Linear layer for feature transformation.
    :ivar dropout: Dropout layer for regularization.
    :ivar out_proj: Output projection layer.
    """

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        """
        Forward pass of the regression head.

        :param features: Input features from wav2vec model.
        :param kwargs: Additional keyword arguments.
        :return: Transformed feature representations.
        """

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    """
    Speech emotion analysis model based on wav2vec2.

    :ivar wav2vec2: Base wav2vec2 model for feature extraction.
    :ivar classifier: Regression head for emotion prediction.
    """

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
        """
        Forward pass of the emotion model.

        :param input_values: Input audio features.
        :return: Tuple of (hidden_states, logits).
        """

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

class Wav2Vec2ADV(PipelineComponent):
    """
    Component for predicting arousal, dominance, and valence from speech using wav2vec2.

    :ivar model: Loaded emotion prediction model.
    :ivar tokenizer: Wav2vec2 tokenizer for processing audio input.
    :ivar device: Device for model computation (CPU/GPU).
    :ivar sampling_rate: Audio sampling rate for processing.
    """
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
        """
        Process raw audio signal to predict emotion dimensions.

        :param x: Raw audio signal array.
        :param sampling_rate: Sampling rate of the audio.
        :return: Array of predicted arousal, dominance, and valence values.
        """

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = self.processor(x, sampling_rate=sampling_rate, return_tensors="pt")
        y = y['input_values'][0]
        y = y.to(self.device)

        # run through model
        with torch.no_grad():
            y = self.model(y.unsqueeze(0))[1]  # Add batch dimension for model

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y

    def process_item(self, f, input_column):
        """
        Process a single audio file for emotion prediction.

        :param f: Path to the audio file.
        :param input_column: Name of the input column.
        :return: DataFrame with predicted emotion dimensions.
        """
        try:
            # Loading the audio file
            audio, rate = librosa.load(f, sr=self.sampling_rate)
            arousal, dominance, valence = self.process_func(audio, rate)[0]

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

        Wav2Vec2ADV.cleanup_softlinks()

        return ComponentPayload(metadata=payload_metadata, df=payload_df)

    @staticmethod
    def cleanup_softlinks():
        """
        Clean up temporary softlinks created during processing.
        """
        for link in os.listdir():
            if '.wav' in link and os.path.islink(link):
                os.unlink(link)