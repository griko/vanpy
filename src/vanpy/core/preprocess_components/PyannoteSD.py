import os
import time
from typing import List, Dict, Any
import pandas as pd
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment


class PyannoteSD(BaseSegmenterComponent):
    """Pyannote speaker diarization model."""

    def __init__(self, yaml_config: YAMLObject):
        """Initialize the PyannoteSD class."""
        super().__init__(component_type='preprocessing', component_name='pyannote_sd', yaml_config=yaml_config)
        self.ACCESS_TOKEN = self.config.get('huggingface_ACCESS_TOKEN')
        if self.ACCESS_TOKEN is None:
            raise KeyError(f'You need to pass huggingface_ACCESS_TOKEN to use {self.component_name} model')
        self.skip_overlap = self.config.get('skip_overlap', False)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')

    def load_model(self):
        """Load the pretrained Pyannote speaker diarization model."""
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
        Get voice segments from the given audio file.

        Args:
            audio_file (str): Path to the audio file.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing segment information.
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

    def process_item(self, audio_file: str, p_df: pd.DataFrame, processed_path: str, input_column: str,
                     output_dir: str) -> pd.DataFrame:
        """
        Process a single audio file for speaker diarization.

        Args:
            audio_file (str): Path to the audio file.
            p_df (pd.DataFrame): DataFrame to store processed data.
            processed_path (str): Column name for processed file paths.
            input_column (str): Column name for input file paths.
            output_dir (str): Directory to save processed audio segments.

        Returns:
            pd.DataFrame: Updated DataFrame with processed data.
        """
        t_start_segmentation = time.time()
        segments = self.get_voice_segments(audio_file)
        t_end_segmentation = time.time()

        for i, segment in enumerate(segments):
            output_path = cut_segment(audio_file, output_dir=output_dir, segment=(segment["start"], segment["stop"]),
                                      segment_id=i, separator=self.segment_name_separator,
                                      keep_only_first_segment=False)

            f_d = {
                processed_path: [output_path],
                input_column: [audio_file],
                self.classification_column_name: [segment["label"]]
            }
            self.add_segment_metadata(f_d, segment["start"], segment["stop"])
            self.add_performance_metadata(f_d, t_start_segmentation, t_end_segmentation)

            f_df = pd.DataFrame.from_dict(f_d)
            p_df = pd.concat([p_df, f_df], ignore_index=True)

        return p_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process the input payload for speaker diarization.

        Args:
            input_payload (ComponentPayload): Input data to be processed.

        Returns:
            ComponentPayload: Processed data.
        """
        if not hasattr(self, 'model'):
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        processed_path = self.get_processed_path()
        p_df = pd.DataFrame()
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(p_df, paths_list, processed_path,
                                                                                   input_column, output_dir)
        metadata = self.enhance_metadata(metadata)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)

        self.config['records_count'] = len(paths_list)

        p_df = self.process_with_progress(paths_list, metadata, self.process_item, p_df, processed_path, input_column,
                                          output_dir)

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)