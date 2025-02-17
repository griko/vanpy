import time
from typing import Union

from yaml import YAMLObject
import pandas as pd
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import cut_segment, create_dirs_if_not_exist


class SileroVAD(BaseSegmenterComponent):
    """
    Voice Activity Detection component using Silero VAD model.

    Segments audio files into voice segments using Silero's state-of-the-art
    voice activity detection model.

    :ivar model: Loaded Silero VAD model instance.
    :ivar utils: Silero utility functions for audio processing.
    :ivar sampling_rate: Target sampling rate for audio processing.
    :ivar params: Model parameters from configuration.
    :ivar keep_only_first_segment: Whether to keep only the first detected segment.
    """
    model = None
    utils = None
    sampling_rate: int

    def __init__(self, yaml_config: YAMLObject):
        """
        Initializes the SileroVAD class and parses the configuration parameters
        :param yaml_config: A YAMLObject containing the configuration for the pipeline
        """
        super().__init__(component_type='preprocessing', component_name='silero_vad',
                         yaml_config=yaml_config)
        self.params = self.config.get('model_params', {})
        self.sampling_rate = self.config.get('sampling_rate', 16000)
        self.keep_only_first_segment = self.config.get('keep_only_first_segment', False)

    def load_model(self):
        """
        Load Silero VAD model and utility functions.
        
        Downloads model from torch hub and configures for GPU if available.
        """
        import torch
        torch.hub.set_dir('pretrained_models/')
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',
                                                force_reload=False)
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

    def process_item(self, f, processed_path, input_column, output_dir) -> pd.DataFrame:
        """
        Process a single audio file for voice activity detection.

        :param f: Path to the audio file.
        :param processed_path: Column name for processed file paths.
        :param input_column: Column name for input file paths.
        :param output_dir: Directory to save processed segments.
        :return: DataFrame containing segment information.
        """
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = self.utils

        t_start_segmentation = time.time()
        wav = read_audio(f, sampling_rate=self.sampling_rate)
        v_segments = [(x['start'] / self.sampling_rate, x['end'] / self.sampling_rate)
                      for x in get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate, **self.params)]
        t_end_segmentation = time.time()

        if not v_segments:
            return pd.DataFrame({
                processed_path: [None],
                input_column: [f]
            })

        f_df = pd.DataFrame()  # Initialize a new DataFrame for each file
        for i, segment in enumerate(v_segments):
            output_path = cut_segment(f, output_dir=output_dir, segment=segment, segment_id=i,
                                      separator=self.segment_name_separator,
                                      keep_only_first_segment=self.keep_only_first_segment)
            s_d = {processed_path: [output_path], input_column: [f]}
            self.add_segment_metadata(s_d, segment[0], segment[1])
            self.add_performance_metadata(s_d, t_start_segmentation, t_end_segmentation)
            s_df = pd.DataFrame.from_dict(s_d)
            f_df = pd.concat([f_df, s_df], ignore_index=True)
            if self.keep_only_first_segment:
                break

        return f_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Processes the input payload by extraction of speech segments identified by SileroVAD.
        The number of segments may be bigger than the number of files in input_payload[metadata['paths_column']].
        It will extract one segment per input path if keep_only_first_segment was set in the configuration.

        :param input_payload: The input payload to process.
        :return: The processed payload.
        """
        if not self.model:
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
