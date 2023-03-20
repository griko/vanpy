from yaml import YAMLObject
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.SegmenterComponent import SegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist, cut_segment
import pandas as pd
import time


class PyannoteSD(SegmenterComponent):
    # Pyannote speaker diarization model
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='pyannote_sd',
                         yaml_config=yaml_config)
        self.ACCESS_TOKEN = self.config['huggingface_ACCESS_TOKEN']
        self.skip_overlap = False if 'skip_overlap' not in self.config else self.config['skip_overlap']
        self.classification_column_name = self.config['classification_column_name'] \
            if 'classification_column_name' in self.config else 'pyannote_diarization_classification'

    def load_model(self):
        from pyannote.audio import Pipeline
        self.model = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                use_auth_token=self.ACCESS_TOKEN,
                                                cache_dir='pretrained_models/pyannote_sd')
        self.model.der_variant['skip_overlap'] = self.skip_overlap

    def get_voice_segments(self, f):
        annotation = self.model(f)
        segments = []
        labels = []
        for i, (v, _, lbl) in enumerate(annotation.itertracks(yield_label=True)):
            start, stop = v
            segments.append((start, stop))
            labels.append(lbl)
        return zip(segments, labels)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].tolist()
        # processed_path = f'{self.get_name()}_processed_path'
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        p_df = pd.DataFrame()
        processed_path, payload_metadata = self.segmenter_create_columns(payload_metadata)
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(p_df, paths_list, processed_path,
                                                                                   input_column, output_dir)
        p_df[self.classification_column_name] = None
        if payload_df.empty:
            ComponentPayload(metadata=payload_metadata, df=payload_df)
        self.config['records_count'] = len(paths_list)

        for j, f in enumerate(paths_list):
            try:
                t_start_segmentation = time.time()
                v_segments = self.get_voice_segments(f)
                segments_count = len(list(v_segments))
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
                end = time.time()
                self.latent_info_log(f'Extracted {segments_count} from {f} in {end - t_start_segmentation} seconds, {j + 1}/{len(paths_list)}', iteration=j)
            except RuntimeError as e:
                self.logger.error(f"An error occurred in {f}, {j + 1}/{len(paths_list)}: {e}")

        payload_metadata['classification_columns'].extend([self.classification_column_name])
        df = pd.merge(left=payload_df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=payload_metadata, df=df)
