import os
import time
import torch
import librosa
from yaml import YAMLObject
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
import pandas as pd


class Wav2Vec2STT(PipelineComponent):
    """
    Speech-to-text component using wav2vec2 model.

    :ivar model: Loaded wav2vec2 model for speech recognition.
    :ivar tokenizer: Tokenizer for processing audio and text.
    :ivar classification_column_name: Name of the output transcription column.
    :ivar sampling_rate: Audio sampling rate for processing.
    """
    model = None
    tokenizer = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the Wav2Vec2STT component.

        :param yaml_config: Configuration parameters for the component.
        """
        super().__init__(component_type='segment_classifier', component_name='wav2vec2stt',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name', f'{self.component_name}_stt')
        self.sampling_rate = self.config.get('sampling_rate', 16000)

    def load_model(self):
        """
        Load the wav2vec2 model and tokenizer.
        """
        self.logger.info("Loading wav2vec 2.0 Speech-To-Text model")
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=self.pretrained_models_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=self.pretrained_models_dir)


    def process_item(self, f, input_column):
        """
        Process a single audio file to extract transcription.

        :param f: Path to the audio file.
        :param input_column: Name of the input column.
        :return: DataFrame with transcription results.
        """
        try:
            # Loading the audio file
            audio, rate = librosa.load(f, sr=self.sampling_rate)
            # Taking an input value
            input_values = self.tokenizer(audio, return_tensors="pt").input_values
            # Storing logits (non-normalized prediction values)
            logits = self.model(input_values).logits
            # Storing predicted ids
            prediction = torch.argmax(logits, dim=-1)
            # Passing the prediction to the tokenizer decode to get the transcription
            transcription = self.tokenizer.batch_decode(prediction)[0]

            # Create a DataFrame to hold the results
            f_df = pd.DataFrame({
                input_column: [f],
                self.classification_column_name: [transcription]
            })

        except (FileNotFoundError, RuntimeError, TypeError, EOFError) as e:
            self.logger.error(f"An error occurred in {f}: {e}")
            f_df = pd.DataFrame({
                input_column: [f],
                self.classification_column_name: [None]
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

        p_df = pd.DataFrame()

        # Define which columns should be in the metadata
        payload_metadata = self.add_performance_column_to_metadata(payload_metadata)
        payload_metadata = self.add_classification_columns_to_metadata(payload_metadata, self.classification_column_name)

        # Call process_with_progress
        p_df = self.process_with_progress(paths_list, payload_metadata, input_column)

        # Merge the processed DataFrame back into the original DataFrame
        payload_df = pd.merge(left=payload_df, right=p_df, how='left', on=input_column)

        Wav2Vec2STT.cleanup_softlinks()

        return ComponentPayload(metadata=payload_metadata, df=payload_df)

    @staticmethod
    def cleanup_softlinks():
        """
        Clean up temporary softlinks created during processing.
        """
        for link in os.listdir():
            if '.wav' in link and os.path.islink(link):
                os.unlink(link)