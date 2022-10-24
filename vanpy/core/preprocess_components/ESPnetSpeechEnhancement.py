import librosa
import soundfile as sf
from yaml import YAMLObject

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PiplineComponent import PipelineComponent
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment
import espnet_model_zoo
from espnet2.bin.enh_inference import SeparateSpeech
import pandas as pd
import time


class ESPnetSpeechEnhancement(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='espnet_se',
                         yaml_config=yaml_config)
        self.sampling_rate = self.config['sampling_rate']
        # self.params = {   # onset/offset activation thresholds
        #                   "onset": self.config['onset'], "offset": self.config['offset'],
        #                   # remove speech regions shorter than that many seconds.
        #                   "min_duration_on": self.config['min_duration_on'],
        #                   # fill non-speech regions shorter than that many seconds.
        #                   "min_duration_off": self.config['min_duration_off']
        #                 }

    def load_model(self):
        from espnet_model_zoo.downloader import ModelDownloader
        d = ModelDownloader()
        cfg = d.download_and_unpack("espnet/Wangyou_Zhang_chime4_enh_train_enh_conv_tasnet_raw")

        from espnet2.bin.enh_inference import SeparateSpeech
        # For models downloaded from GoogleDrive, you can use the following script:
        enh_model_sc = SeparateSpeech(
            train_config=cfg["train_config"],
            model_file=cfg["model_file"],
            # for segment-wise process on long speech
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=4,
            normalize_output_wav=True,
            device="cpu:0",
        )

        self.model = enh_model_sc

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload

        p_df = pd.DataFrame()
        processed_path = f'{self.get_name()}_processed_path'
        metadata['paths_column'] = processed_path
        metadata['all_paths_columns'].append(processed_path)
        file_performance_column_name = ''
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_voice_segments'
            metadata['meta_columns'].extend([file_performance_column_name])

        for f in paths_list:
            try:
                t_start_segmentation = time.time()
                y, sr = librosa.load(f, sr=self.sampling_rate)
                wave = self.model(y[None, ...], sr)
                output_path = f'{output_dir}/{f}.wav'
                sf.write(output_path, wave[0].squeeze(), sr)
                wave[0].squeeze()
                t_end_segmentation = time.time()

                f_d = {processed_path: [output_path], input_column: [f]}
                if self.config['performance_measurement']:
                    f_d[file_performance_column_name] = t_end_segmentation - t_start_segmentation
                f_df = pd.DataFrame.from_dict(f_d)
                p_df = pd.concat([p_df, f_df], ignore_index=True)

            except RuntimeError as err:
                self.logger.error(f"Could not create ESPNet-SE pipline for {f}.\n{err}")

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
