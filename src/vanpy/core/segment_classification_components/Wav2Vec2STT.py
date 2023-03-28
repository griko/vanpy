import time
import torch
import librosa
from yaml import YAMLObject
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent


class Wav2Vec2STT(PipelineComponent):
    model = None
    tokenizer = None
    classification_column_name: str = ''
    pretrained_models_dir: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='wav2vec2stt',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config['classification_column_name']
        self.pretrained_models_dir = self.config['pretrained_models_dir']

    def load_model(self):
        self.logger.info("Loading wav2vec 2.0 Speech-To-Text model")
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=self.pretrained_models_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=self.pretrained_models_dir)

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
        
        stts = []
        performance_metric = []
        sr = self.config['sampling_rate'] if 'sampling_rate' in self.config else 16000
        for j, f in enumerate(paths_list):
            try:
                t_start_transcribing = time.time()
                # Loading the audio file
                audio, rate = librosa.load(f, sr=sr)
                # Taking an input value
                input_values = self.tokenizer(audio, return_tensors="pt").input_values
                # Storing logits (non-normalized prediction values)
                logits = self.model(input_values).logits
                # Storing predicted ids
                prediction = torch.argmax(logits, dim=-1)
                # Passing the prediction to the tokenizer decode to get the transcription
                transcription = self.tokenizer.batch_decode(prediction)[0]
                stts.append(transcription)
                t_end_transcribing = time.time()
                performance_metric.append(t_end_transcribing - t_start_transcribing)
                self.latent_info_log(
                    f'Transcribed {f} in {t_end_transcribing - t_start_transcribing} seconds, {j + 1}/{len(paths_list)}',
                    iteration=j)
            except RuntimeError as e:
                stts.append(None)
                performance_metric.append(float('inf'))
                self.logger.error(f'An error occurred in {f}, {j + 1}/{len(paths_list)}: {e}')

        payload_df[self.classification_column_name] = stts
        payload_metadata['classification_columns'].extend([self.classification_column_name])
        if 'performance_measurement' in self. config and self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_transcription'
            payload_df[file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
