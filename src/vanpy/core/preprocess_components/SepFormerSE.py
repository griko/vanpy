import os
from yaml import YAMLObject
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment, get_audio_files_paths
import time


class SepFormerSE(PipelineComponent):
    """
    Pipeline component for SepFormer speech enhancement.
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
        Loads the SepFormerSeparation pretrained model from huggingface and saves it to self.model.
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

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Processes the input payload by performing speech enhancement using the SepFormer model.

        :param input_payload: The input payload containing the paths to the audio files to be enhanced.
        :return: The output payload containing the paths to the enhanced audio files.
        """
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
                    enhanced = self.model.separate_file(path=f)
                    torchaudio.save(output_file, enhanced[:, :, 0].detach().cpu(), 16000)
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
        SepFormerSE.cleanup_softlinks()
        return ComponentPayload(metadata=payload_metadata, df=payload_df)

    @staticmethod
    def cleanup_softlinks():
        """
        Remove all soft links to .wav files in the current directory.
        """
        for link in os.listdir():
            if '.wav' in link and os.path.islink(link):
                os.unlink(link)
