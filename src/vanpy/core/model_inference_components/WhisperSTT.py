import time
import whisper
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import create_dirs_if_not_exist
import pandas as pd

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

    def process_item(self, f, p_df, input_column, stt_column_name, language_column_name):
        try:
            transcription = self.model.transcribe(f)
            stt = transcription['text']
            language = transcription['language']
        except Exception as e:
            self.logger.error(f'Failed to transcribe {f}: {e}')
            stt = None
            language = None

        f_df = pd.DataFrame({input_column: [f], stt_column_name: [stt], language_column_name: [language]})
        p_df = pd.concat([p_df, f_df], ignore_index=True)
        return p_df

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

        p_df = pd.DataFrame()
        payload_metadata['classification_columns'].extend([self.stt_column_name, self.language_classification_column_name])

        p_df = self.process_with_progress(
            paths_list,
            payload_metadata,
            self.process_item,
            p_df,
            input_column,
            self.stt_column_name,
            self.language_classification_column_name
        )

        payload_df = pd.merge(
            left=payload_df,
            right=p_df,
            how='outer',
            left_on=input_column,
            right_on=input_column
        )

        if self.config.get('performance_measurement', False):
            file_performance_column_name = f'perf_{self.get_name()}_get_transcription'
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
