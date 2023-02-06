import time
import whisper
import torch
import librosa
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import create_dirs_if_not_exist


class WhisperSTT(PipelineComponent):
    model = None
    classification_column_name: str = ''
    pretrained_models_dir: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='openai_whisper_stt',
                         yaml_config=yaml_config)
        self.stt_column_name = self.config['stt_column_name']
        self.language_classification_column_name = self.config['language_classification_column_name'] \
            if 'language_classification_column_name' in self.config else 'whisper_transcript'
        self.pretrained_models_dir = self.config['pretrained_models_dir'] \
            if 'pretrained_models_dir' in self.config else 'pretrained_models/whisper'
        create_dirs_if_not_exist(self.pretrained_models_dir)
        self.model_size = self.config['model_size'] if 'model_size' in self.config else 'small'

    def load_model(self):
        self.logger.info("Loading openai-whisper model")
        self.model = whisper.load_model(self.model_size, download_root=self.pretrained_models_dir)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].tolist()
        self.config['records_count'] = len(paths_list) - 1

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload

        stts = []
        languages = []
        performance_metric = []
        sr = self.config['sampling_rate'] if 'sampling_rate' in self.config else 16000
        for j, f in enumerate(paths_list):
            t_start_transcribing = time.time()
            # Loading the audio file
            # audio, rate = librosa.load(f, sr=sr)
            audio = whisper.load_audio(f, sr=sr)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            _, probs = self.model.detect_language(mel)

            options = whisper.DecodingOptions(fp16=False)
            transcription = whisper.decode(self.model, mel, options)
            stts.append(transcription.text)
            if 'detect_language' in self.config and self.config['detect_language']:
                languages.append(transcription.language)
            t_end_transcribing = time.time()
            performance_metric.append(t_end_transcribing - t_start_transcribing)
            self.latent_info_log(
                f'Transcribed {f} in {t_end_transcribing - t_start_transcribing} seconds, {j + 1}/{len(paths_list)}',
                iteration=j)

        payload_df[self.stt_column_name] = stts
        payload_metadata['classification_columns'].extend([self.stt_column_name])
        if 'detect_language' in self.config and self.config['detect_language']:
            payload_df[self.language_classification_column_name] = languages
            payload_metadata['classification_columns'].extend([self.language_classification_column_name])
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_transcription'
            payload_df[file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
