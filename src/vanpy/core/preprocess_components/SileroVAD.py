import time
from yaml import YAMLObject
import pandas as pd
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.SegmenterComponent import SegmenterComponent
from vanpy.utils.utils import cut_segment, create_dirs_if_not_exist


class SileroVAD(SegmenterComponent):
    model = None
    utils = None
    sampling_rate: int

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='silero_vad',
                         yaml_config=yaml_config)
        self.params = self.config.get('model_params', {})
        self.sampling_rate = self.config.get('sampling_rate', 16000)
        self.keep_only_first_segment = self.config.get('keep_only_first_segment', False)

    def load_model(self):
        import torch
        torch.hub.set_dir('pretrained_models/')
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',
                                                force_reload=False)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        p_df = pd.DataFrame()
        processed_path, metadata = self.segmenter_create_columns(metadata)
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(p_df, paths_list, processed_path,
                                                                                   input_column, output_dir)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)
        self.config['records_count'] = len(paths_list)

        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = self.utils

        for j, f in enumerate(paths_list):
            try:
                t_start_segmentation = time.time()
                wav = read_audio(f, sampling_rate=self.sampling_rate)
                # get speech timestamps from full audio file
                v_segments = [(x['start'] / self.sampling_rate, x['end'] / self.sampling_rate) for x in
                              get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate, **self.params)]
                t_end_segmentation = time.time()
                for i, segment in enumerate(v_segments):
                    output_path = cut_segment(f, output_dir=output_dir, segment=segment, segment_id=i,
                                              separator=self.segment_name_separator,
                                              keep_only_first_segment=self.keep_only_first_segment)
                    f_d = {processed_path: [output_path], input_column: [f]}
                    self.add_segment_metadata(f_d, segment[0], segment[1])
                    self.add_performance_metadata(f_d, t_start_segmentation, t_end_segmentation)
                    f_df = pd.DataFrame.from_dict(f_d)
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
                    if self.keep_only_first_segment:
                        break
                self.latent_info_log(
                    f'Extracted {len(v_segments)} from {f} in {t_end_segmentation - t_start_segmentation} seconds, {j + 1}/{len(paths_list)}', iteration=j)

            except RuntimeError as e:
                self.logger.error(f"An error occurred in {f}, {j + 1}/{len(paths_list)}: {e}")
            self.save_intermediate_payload(j, ComponentPayload(metadata=metadata, df=p_df))

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
