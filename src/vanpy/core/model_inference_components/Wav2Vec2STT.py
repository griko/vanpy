import time
import torch
import librosa
from yaml import YAMLObject
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
import pandas as pd


class Wav2Vec2STT(PipelineComponent):
    model = None
    tokenizer = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='wav2vec2stt',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name', f'{self.component_name}_stt')
        self.sampling_rate = self.config.get('sampling_rate', 16000)

    def load_model(self):
        self.logger.info("Loading wav2vec 2.0 Speech-To-Text model")
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=self.pretrained_models_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=self.pretrained_models_dir)


    def process_item(self, f, p_df, input_column):
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

        except (RuntimeError, TypeError, EOFError) as e:
            self.logger.error(f"An error occurred in {f}: {e}")
            f_df = pd.DataFrame({
                input_column: [f],
                self.classification_column_name: [None]
            })

        p_df = pd.concat([p_df, f_df], ignore_index=True)

        return p_df

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
        paths_list = payload_df[input_column].tolist()
        self.config['records_count'] = len(paths_list)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload

        p_df = pd.DataFrame()

        # Define which columns should be in the metadata
        payload_metadata = self.add_performance_column_to_metadata(payload_metadata)
        payload_metadata = self.add_classification_columns_to_metadata(payload_metadata, self.classification_column_name)

        # Call process_with_progress
        p_df = self.process_with_progress(paths_list, payload_metadata, self.process_item, p_df, input_column)

        # Merge the processed DataFrame back into the original DataFrame
        payload_df = pd.merge(left=payload_df, right=p_df, how='outer', left_on=input_column, right_on=input_column)

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
