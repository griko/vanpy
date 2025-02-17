import time
import whisper
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import create_dirs_if_not_exist
import pandas as pd

class WhisperSTT(PipelineComponent):
    """
    Speech-to-text component using OpenAI's Whisper model.

    :ivar model: Loaded Whisper model instance.
    :ivar stt_column_name: Name of the transcription output column.
    :ivar language_classification_column_name: Name of the detected language column.
    :ivar model_size: Size of the Whisper model to use.
    """
    model = None
    classification_column_name: str = ''

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the WhisperSTT component.

        :param yaml_config: Configuration parameters for the component.
        """
        super().__init__(component_type='segment_classifier', component_name='openai_whisper_stt',
                         yaml_config=yaml_config)
        self.stt_column_name = self.config.get('stt_column_name', 'whisper_transcript')
        self.language_classification_column_name = self.config.get('language_classification_column_name', 'whisper_language')
        create_dirs_if_not_exist(self.pretrained_models_dir)
        self.model_size = self.config.get('model_size', 'small')

    def load_model(self):
        """
        Load the Whisper model and move to appropriate device.
        """
        import torch
        self.logger.info("Loading openai-whisper speech-to-text model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = whisper.load_model(self.model_size, download_root=self.pretrained_models_dir).to(device)
        self.model.eval()
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

    def process_item(self, f, input_column, stt_column_name, language_column_name):
        """
        Process a single audio file for transcription and language detection.

        :param f: Path to the audio file.
        :param input_column: Name of the input column.
        :param stt_column_name: Name of the transcription output column.
        :param language_column_name: Name of the language detection column.
        :return: DataFrame with transcription and language detection results.
        """
        try:
            transcription = self.model.transcribe(f)
            stt = transcription['text']
            language = transcription['language']
            return pd.DataFrame({
                input_column: [f],
                stt_column_name: [stt],
                language_column_name: [language]
            })
        except Exception as e:
            self.logger.error(f'Failed to transcribe {f}: {e}')
            return pd.DataFrame({input_column: [f]})

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].dropna().tolist()

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload

        payload_metadata['classification_columns'].extend([self.stt_column_name, self.language_classification_column_name])

        p_df = self.process_with_progress(
            paths_list,
            payload_metadata,
            input_column,
            self.stt_column_name,
            self.language_classification_column_name
        )

        payload_df = pd.merge(
            left=payload_df,
            right=p_df,
            how='left',
            on=input_column,
            # validate='1:m'
        )

        if self.config.get('performance_measurement', False):
            file_performance_column_name = f'perf_{self.get_name()}_get_transcription'
            payload_metadata['meta_columns'].extend([file_performance_column_name])

        return ComponentPayload(metadata=payload_metadata, df=payload_df)
