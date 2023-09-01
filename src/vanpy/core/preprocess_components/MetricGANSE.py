import os

import pandas as pd
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment, get_audio_files_paths
import time
import torch


class MetricGANSE(BaseSegmenterComponent):
    # MagicGAN speech enhancement component
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='metricgan_se',
                         yaml_config=yaml_config)

    def load_model(self):
        from speechbrain.pretrained import SpectralMaskEnhancement
        if torch.cuda.is_available():
            self.model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",
                                                          savedir="pretrained_models/metricgan-plus-voicebank",
                                                          run_opts={"device": "cuda"})
        else:
            self.model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",
                                                              savedir="pretrained_models/metricgan-plus-voicebank",)
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

    def process_item(self, f, p_df, processed_path, input_column, output_dir):
        import torch
        import torchaudio

        output_file = f'{output_dir}/{f.split("/")[-1]}'
        t_start_segmentation = time.time()

        # Load and add fake batch dimension
        noisy = self.model.load_audio(f).unsqueeze(0)

        # Add relative length tensor
        enhanced = self.model.enhance_batch(noisy, lengths=torch.tensor([1.]))

        # Saving enhanced signal on disk
        torchaudio.save(output_file, enhanced.cpu(), self.config['sampling_rate'])
        t_end_segmentation = time.time()

        f_d = {processed_path: [output_file], input_column: [f]}
        self.add_performance_metadata(f_d, t_start_segmentation, t_end_segmentation)
        f_df = pd.DataFrame.from_dict(f_d)
        p_df = pd.concat([p_df, f_df], ignore_index=True)
        return p_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
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
        metadata = self.add_processed_path_to_metadata(self.get_processed_path(), metadata)
        metadata = self.add_performance_column_to_metadata(metadata)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)
        self.config['records_count'] = len(paths_list)

        p_df = self.process_with_progress(paths_list, metadata, self.process_item, p_df, processed_path,
                                          input_column, output_dir)

        MetricGANSE.cleanup_softlinks()
        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)

    @staticmethod
    def cleanup_softlinks():
        for link in os.listdir():
            if '.wav' in link and os.path.islink(link):
                os.unlink(link)
