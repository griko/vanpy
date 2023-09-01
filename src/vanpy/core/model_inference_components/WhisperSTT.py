import time
import whisper
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import create_dirs_if_not_exist


class WhisperSTT(PipelineComponent):
    model = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='openai_whisper_stt',
                         yaml_config=yaml_config)
        self.stt_column_name = self.config.get('stt_column_name', 'whisper_transcript')
        self.language_classification_column_name = self.config.get('language_classification_column_name', 'whisper_language')
        create_dirs_if_not_exist(self.pretrained_models_dir)
        self.model_size = self.config.get('model_size', 'small')

    def load_model(self):
        import torch
        self.logger.info("Loading openai-whisper speech-to-text model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = whisper.load_model(self.model_size, download_root=self.pretrained_models_dir).to(device)
        self.model.eval()
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

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
        languages = []
        performance_metric = []
        for j, f in enumerate(paths_list):
            try:
                t_start_transcribing = time.time()
                transcription = self.model.transcribe(f)
                stts.append(transcription['text'])
                languages.append(transcription['language'])
                t_end_transcribing = time.time()
                performance_metric.append(t_end_transcribing - t_start_transcribing)
                self.latent_info_log(
                    f'Transcribed {f} in {t_end_transcribing - t_start_transcribing} seconds, {j + 1}/{len(paths_list)}',
                    iteration=j)
            except Exception as e:
                self.logger.error(f'Failed to transcribe {f}, {j + 1}/{len(paths_list)}: {e}')
                stts.append(None)
                languages.append(None)
                performance_metric.append(None)

        payload_df[self.stt_column_name] = stts
        payload_metadata['classification_columns'].extend([self.stt_column_name])
        if self.config.get('detect_language', False):
            payload_df[self.language_classification_column_name] = languages
            payload_metadata['classification_columns'].extend([self.language_classification_column_name])
        if self.config.get('performance_measurement', False):
            file_performance_column_name = f'perf_{self.get_name()}_get_transcription'
            payload_df[file_performance_column_name] = performance_metric
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
