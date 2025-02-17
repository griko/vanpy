import os

import pandas as pd
from yaml import YAMLObject
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment, get_audio_files_paths
import time


class SepFormerSE(BaseSegmenterComponent):
    """
    Speech enhancement component using SepFormer model.

    Enhances speech quality by separating speech from noise using the
    SepFormer model from SpeechBrain.

    :ivar model: Loaded SepFormer enhancement model instance.
    """
    model = None

    def __init__(self, yaml_config: YAMLObject):
        """
        Initializes the SepFormerSE component.

        :param yaml_config: A YAMLObject containing the configuration parameters for the component.
        """
        super().__init__(component_type='preprocessing', component_name='sepformer_se',
                         yaml_config=yaml_config)

    def load_model(self):
        """
        Load SepFormer model from SpeechBrain.

        Downloads pretrained model and moves to GPU if available.
        Sets model to evaluation mode.
        """
        import torch
        from speechbrain.pretrained import SepformerSeparation
        if torch.cuda.is_available():
            self.model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wham16k-enhancement",
                                                          savedir="pretrained_models/sepformer-wham16k-enhancement",
                                                          run_opts={"device": "cuda"})
        else:
            self.model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wham16k-enhancement",
                                                              savedir="pretrained_models/sepformer-wham16k-enhancement",)
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

    def process_item(self, f, processed_path, input_column, output_dir):
        """
        Process a single audio file for speech enhancement.

        :param f: Path to the audio file.
        :param processed_path: Column name for processed file paths.
        :param input_column: Column name for input file paths.
        :param output_dir: Directory to save enhanced audio.
        :return: DataFrame containing enhanced audio information.
        """
        import torchaudio
        output_file = f'{output_dir}/{f.split("/")[-1]}'
        t_start_segmentation = time.time()
        enhanced = self.model.separate_file(path=f)
        torchaudio.save(output_file, enhanced[:, :, 0].detach().cpu(), 16000)
        t_end_segmentation = time.time()
        f_d = {processed_path: [output_file], input_column: [f]}
        self.add_performance_metadata(f_d, t_start_segmentation, t_end_segmentation)
        f_df = pd.DataFrame.from_dict(f_d)
        return f_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Processes the input payload by performing speech enhancement using the SepFormer model.

        :param input_payload: The input payload containing the paths to the audio files to be enhanced.
        :return: The output payload containing the paths to the enhanced audio files.
        """
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].dropna().tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        processed_path = self.get_processed_path()
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(paths_list, processed_path,
                                                                                   input_column, output_dir)
        metadata = self.add_processed_path_to_metadata(self.get_processed_path(), metadata)
        metadata = self.add_performance_column_to_metadata(metadata)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='left', on=input_column)
            return ComponentPayload(metadata=metadata, df=df)

        p_df = self.process_with_progress(paths_list, metadata, processed_path,
                                          input_column, output_dir)

        SepFormerSE.cleanup_softlinks()
        df = pd.merge(left=df, right=p_df, how='left', on=input_column)
        return ComponentPayload(metadata=metadata, df=df)

    @staticmethod
    def cleanup_softlinks():
        """
        Remove all soft links to .wav files in the current directory.
        """
        for link in os.listdir():
            if '.wav' in link and os.path.islink(link):
                os.unlink(link)
