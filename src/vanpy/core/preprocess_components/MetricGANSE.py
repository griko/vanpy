import os
from yaml import YAMLObject
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment, get_audio_files_paths
import time
import torch


class MetricGANSE(PipelineComponent):
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

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        import torch
        import torchaudio
        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].tolist()
        processed_path = f'{self.get_name()}_processed_path'
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        if payload_df.empty:
            ComponentPayload(metadata=payload_metadata, df=payload_df)
        self.config['records_count'] = len(paths_list)

        processed_paths_list, existing_files = [], []
        if self.config.get('overwrite', False):
            existing_files = get_audio_files_paths(output_dir)
        for j, f in enumerate(paths_list):
            try:
                output_file = f'{output_dir}/{f.split("/")[-1]}'

                if output_file not in existing_files:
                    t_start_segmentation = time.time()
                    # Load and add fake batch dimension
                    noisy = self.model.load_audio(f).unsqueeze(0)
                    # Add relative length tensor
                    enhanced = self.model.enhance_batch(noisy, lengths=torch.tensor([1.]))
                    # Saving enhanced signal on disk

                    torchaudio.save(output_file, enhanced.cpu(), 16000)
                    end = time.time()
                    self.latent_info_log(
                        f'Enhanced {f} in {end - t_start_segmentation} seconds, {j + 1}/{len(paths_list)}',
                        iteration=j)
                else:
                    self.latent_info_log(f'Skipping enhancement for {f}, already exists, {j + 1}/{len(paths_list)}', iteration=j)
                processed_paths_list.append(output_file)
            except RuntimeError as e:
                processed_paths_list.append(None)
                self.logger.error(f"An error occurred in {f}, {j + 1}/{len(paths_list)}: {e}")

        payload_df[processed_path] = processed_paths_list
        MetricGANSE.cleanup_softlinks()
        return ComponentPayload(metadata=payload_metadata, df=payload_df)

    @staticmethod
    def cleanup_softlinks():
        for link in os.listdir():
            if '.wav' in link and os.path.islink(link):
                os.unlink(link)
