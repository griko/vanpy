from typing import List, Tuple
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment
import pandas as pd
import time


class PyannoteVAD(BaseSegmenterComponent):
    """
    Voice Activity Detection component using Pyannote models.

    Segments audio files into voice segments using Pyannote's VAD model.
    Supports GPU acceleration if available.

    :ivar model: Loaded Pyannote VAD model instance.
    :ivar params: Model parameters from configuration.
    :ivar ACCESS_TOKEN: HuggingFace access token for model download.
    :ivar keep_only_first_segment: Whether to keep only the first detected segment.
    """
    model = None

    def __init__(self, yaml_config: YAMLObject):
        """
        Initializes the PyannoteVAD class and parses the configuration parameters
        """
        super().__init__(component_type='preprocessing', component_name='pyannote_vad',
                         yaml_config=yaml_config)
        self.params = self.config.get('model_params', {})
        self.ACCESS_TOKEN = self.config.get('huggingface_ACCESS_TOKEN', None)
        if self.ACCESS_TOKEN is None:
            raise KeyError(f'You need to pass huggingface_ACCESS_TOKEN to use {self.component_name} model')
        self.keep_only_first_segment = self.config.get('keep_only_first_segment', False)

    def load_model(self):
        """
        Load Pyannote VAD model and initialize with configuration parameters.

        Uses HuggingFace access token to download pretrained model.
        Automatically selects GPU if available.
        """
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
        import torch
        model = Model.from_pretrained("pyannote/segmentation",
                                      use_auth_token=self.ACCESS_TOKEN,
                                      cache_dir='pretrained_models/pyannote_vad')
        self.model = VoiceActivityDetection(segmentation=model)
        self.model.instantiate(self.params)
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.device_count() > 0 else "CPU"}')

    def get_voice_segments(self, f: str) -> List[Tuple[float, float]]:
        """
        Extract voice segments from an audio file.

        :param f: Path to the audio file.
        :return: List of (start_time, end_time) tuples in seconds.
        """
        annotation = self.model(f)
        segments = []
        for i, v in enumerate(annotation.itersegments()):
            start, stop = v
            segments.append((start, stop))
        return segments

    def process_item(self, f, processed_path, input_column, output_dir):
        """
        Process a single audio file for voice activity detection.

        :param f: Path to the audio file.
        :param processed_path: Column name for processed file paths.
        :param input_column: Column name for input file paths.
        :param output_dir: Directory to save processed segments.
        :return: DataFrame containing segment information.
        """
        t_start_segmentation = time.time()
        v_segments = self.get_voice_segments(f)
        t_end_segmentation = time.time()

        if not v_segments:
            return pd.DataFrame({
                processed_path: [None],
                input_column: [f]
            })

        f_df = pd.DataFrame()

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
        Process audio files for voice activity detection.

        :param input_payload: Input payload containing audio files and metadata.
        :return: Output payload containing voice activity detection results.
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

