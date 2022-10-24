import time
from yaml import YAMLObject
import torch
import pandas as pd

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PiplineComponent import PipelineComponent
from vanpy.utils.utils import cut_segment, create_dirs_if_not_exist


class SileroVAD(PipelineComponent):
    model = None
    utils = None
    sampling_rate: int

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='silero_vad',
                         yaml_config=yaml_config)
        self.sampling_rate = self.config['sampling_rate']

    def load_model(self):
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

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

        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = self.utils

        p_df = pd.DataFrame()
        processed_path = f'{self.get_name()}_processed_path'
        metadata['paths_column'] = processed_path
        metadata['all_paths_columns'].append(processed_path)
        segment_start_column_name = segment_stop_column_name = ''
        if self.config['add_segment_metadata']:
            segment_start_column_name = f'{self.get_name()}_segment_start'
            segment_stop_column_name = f'{self.get_name()}_segment_stop'
            metadata['meta_columns'].extend([segment_start_column_name, segment_stop_column_name])
        file_performance_column_name = ''
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_voice_segments'
            metadata['meta_columns'].extend([file_performance_column_name])
        for f in paths_list:
            try:
                t_start_segmentation = time.time()
                wav = read_audio(f, sampling_rate=self.sampling_rate)
                # get speech timestamps from full audio file
                v_segments = [(x['start'] / self.sampling_rate, x['end'] / self.sampling_rate) for x in
                              get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate)]
                t_end_segmentation = time.time()
                for i, segment in enumerate(v_segments):
                    output_path = cut_segment(f, output_dir=output_dir, segment=segment, segment_id=i)
                    f_d = {processed_path: [output_path], input_column: [f]}
                    if self.config['add_segment_metadata']:
                        f_d[segment_start_column_name] = [segment[0]]
                        f_d[segment_stop_column_name] = [segment[1]]
                    if self.config['performance_measurement']:
                        f_d[file_performance_column_name] = t_end_segmentation - t_start_segmentation
                    f_df = pd.DataFrame.from_dict(f_d)
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
                self.logger.info(f'Extracted {len(v_segments)} from {f} in {t_end_segmentation - t_start_segmentation} seconds')
            except RuntimeError as err:
                self.logger.error(f"Could not create VAD pipline for {f} with pyannote.\n{err}")

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
