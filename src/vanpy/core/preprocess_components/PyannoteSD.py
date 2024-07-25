import os

import yaml
from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment
import pandas as pd
import time


class PyannoteSD(BaseSegmenterComponent):
    # Pyannote speaker diarization model
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='pyannote_sd',
                         yaml_config=yaml_config)
        self.ACCESS_TOKEN = self.config.get('huggingface_ACCESS_TOKEN', None)
        if self.ACCESS_TOKEN is None:
            raise KeyError(f'You need to pass huggingface_ACCESS_TOKEN to use {self.component_name} model')
        self.skip_overlap = self.config.get('skip_overlap', False)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')

    def load_model(self):
        from pyannote.audio import Pipeline
        import torch
        if 'hparams' in self.config:
            yaml.dump(self.config['hparams'], open('pyannote_sd.yaml', 'w'), default_flow_style=False)
        if os.path.exists('pyannote_sd.yaml'):
            self.model = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                  use_auth_token=self.ACCESS_TOKEN, hparams_file='pyannote_sd.yaml',
                                                  cache_dir='pretrained_models/pyannote_sd')
        else:
            self.model = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                  use_auth_token=self.ACCESS_TOKEN,
                                                  cache_dir='pretrained_models/pyannote_sd')
        self.model.der_variant['skip_overlap'] = self.skip_overlap
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.device_count() > 0 else "CPU"}')

    def get_voice_segments(self, f):
        try:
            annotation = self.model(f)
            segments = []
            labels = []
            for i, (v, _, lbl) in enumerate(annotation.itertracks(yield_label=True)):
                start, stop = v
                segments.append((start, stop))
                labels.append(lbl)
            return zip(segments, labels)
        except ValueError as e:
            self.logger.error(f'Error in {f}: {e}')
            return []

    def process_item(self, f, p_df, processed_path, input_column, output_dir):
        t_start_segmentation = time.time()
        v_segments = self.get_voice_segments(f)
        t_end_segmentation = time.time()

        for i, (segment, label) in enumerate(v_segments):
            output_path = cut_segment(f, output_dir=output_dir, segment=segment, segment_id=i,
                                      separator=self.segment_name_separator,
                                      keep_only_first_segment=False)
            f_d = {processed_path: [output_path], input_column: [f], self.classification_column_name: [label]}
            self.add_segment_metadata(f_d, segment[0], segment[1])
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
        metadata = self.enhance_metadata(metadata)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
            return ComponentPayload(metadata=metadata, df=df)
        self.config['records_count'] = len(paths_list)

        p_df = self.process_with_progress(paths_list, metadata, self.process_item, p_df, processed_path, input_column, output_dir)

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
