import os
import time
from typing import List, Dict, Any
import pandas as pd
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment


class PyannoteSD(BaseSegmenterComponent):
    """
    Speaker diarization component using Pyannote's speaker diarization pipeline.

    Performs speaker diarization on audio files using Pyannote's state-of-the-art
    pipeline, identifying different speakers and their speaking segments.

    :ivar model: Loaded Pyannote speaker diarization pipeline instance.
    :ivar ACCESS_TOKEN: HuggingFace access token for model download.
    :ivar skip_overlap: Whether to skip overlapping speech segments.
    :ivar classification_column_name: Name of the output classification column.
    :ivar keep_only_first_segment: Whether to keep only the first detected segment.
    """

    def __init__(self, yaml_config: YAMLObject):
        """Initialize the PyannoteSD class."""
        super().__init__(component_type='preprocessing', component_name='pyannote_sd', yaml_config=yaml_config)
        self.ACCESS_TOKEN = self.config.get('huggingface_ACCESS_TOKEN')
        if self.ACCESS_TOKEN is None:
            raise KeyError(f'You need to pass huggingface_ACCESS_TOKEN to use {self.component_name} model')
        self.skip_overlap = self.config.get('skip_overlap', False)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')
        self.keep_only_first_segment = self.config.get('keep_only_first_segment', False)

    def load_model(self):
        """
        Load and configure the Pyannote speaker diarization pipeline.

        Downloads the pretrained model using HuggingFace access token and configures
        it with custom hyperparameters if provided. Automatically selects GPU if available.

        Supports custom hyperparameters through 'hparams' configuration section,
        which will be saved to 'pyannote_sd.yaml' if provided.
        """
        from pyannote.audio import Pipeline
        import torch
        import yaml

        if 'hparams' in self.config:
            yaml.dump(self.config['hparams'], open('pyannote_sd.yaml', 'w'), default_flow_style=False)

        model_path = "pyannote/speaker-diarization@2.1"
        cache_dir = 'pretrained_models/pyannote_sd'

        if os.path.exists('pyannote_sd.yaml'):
            self.model = Pipeline.from_pretrained(model_path, use_auth_token=self.ACCESS_TOKEN,
                                                  hparams_file='pyannote_sd.yaml', cache_dir=cache_dir)
        else:
            self.model = Pipeline.from_pretrained(model_path, use_auth_token=self.ACCESS_TOKEN, cache_dir=cache_dir)

        self.model.der_variant['skip_overlap'] = self.skip_overlap
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.device_count() > 0 else "CPU"}')

    def get_voice_segments(self, audio_file: str) -> List[Dict[str, Any]]:
        """
        Extract speaker segments from an audio file.

        Processes the audio file through the diarization pipeline to identify
        different speakers and their speaking segments.

        :param audio_file: Path to the audio file to process.
        :return: List of dictionaries containing segment information:
                - "start": Start time in seconds
                - "stop": End time in seconds
                - "label": Speaker label
        :raises ValueError: If audio file processing fails.
        """
        try:
            annotation = self.model(audio_file)
            segments = []
            for segment, _, label in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": segment.start,
                    "stop": segment.end,
                    "label": label
                })
            return segments
        except ValueError as e:
            self.logger.error(f'Error in {audio_file}: {e}')
            return []

    def process_item(self, audio_file: str, processed_path: str, input_column: str,
                     output_dir: str) -> pd.DataFrame:
        """
        Process a single audio file for speaker diarization.

        Extracts speaker segments and creates separate audio files for each segment,
        with timing and speaker label information.

        :param audio_file: Path to the audio file to process.
        :param processed_path: Column name for processed file paths.
        :param input_column: Column name for input file paths.
        :param output_dir: Directory to save processed segments.
        :return: DataFrame containing segment information and file paths.
        """
        t_start_segmentation = time.time()
        segments = self.get_voice_segments(audio_file)
        t_end_segmentation = time.time()

        if not segments:
            return pd.DataFrame({
                processed_path: [None],
                input_column: [audio_file]
            })

        f_df = pd.DataFrame()
        for i, segment in enumerate(segments):
            output_path = cut_segment(audio_file, output_dir=output_dir, segment=(segment["start"], segment["stop"]),
                                      segment_id=i, separator=self.segment_name_separator,
                                      keep_only_first_segment=self.keep_only_first_segment)

            s_d = {
                processed_path: [output_path],
                input_column: [audio_file],
                self.classification_column_name: [segment["label"]]
            }
            self.add_segment_metadata(s_d, segment["start"], segment["stop"])
            self.add_performance_metadata(s_d, t_start_segmentation, t_end_segmentation)

            s_df = pd.DataFrame.from_dict(s_d)
            f_df = pd.concat([f_df, s_df], ignore_index=True)

            if self.keep_only_first_segment:
                break
        return f_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process a batch of audio files for speaker diarization.

        Handles batch processing of audio files, creating separate segments for
        different speakers while maintaining metadata and timing information.
        Supports incremental processing by checking for existing outputs.

        :param input_payload: Input payload containing audio files and metadata.
        :return: Output payload containing diarization results and segment files.
        """
        if not hasattr(self, 'model'):
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].dropna().tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        processed_path = self.get_processed_path()
        metadata = self.enhance_metadata(metadata)

        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(paths_list, processed_path,
                                                                                   input_column, output_dir)
        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
        else:
            fp_df = self.process_with_progress(paths_list, metadata, processed_path, input_column, output_dir)
            p_df = pd.concat([p_df, fp_df], ignore_index=True)

        df = pd.merge(left=df, right=p_df, how='outer', on=input_column)
        return ComponentPayload(metadata=metadata, df=df)