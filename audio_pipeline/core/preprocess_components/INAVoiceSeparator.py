from yaml import YAMLObject
from audio_pipeline.core.PiplineComponent import PipelineComponent, ComponentPayload
from audio_pipeline.utils.utils import create_dirs_if_not_exist, cut_segment
from inaSpeechSegmenter import Segmenter
import pandas as pd
import time


class INAVoiceSeparator(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='ina_speech_segmenter',
                         yaml_config=yaml_config)

    def load_model(self):
        self.model = Segmenter(vad_engine=self.config['vad_engine'])

    @classmethod
    def get_voice_segments(cls, segmentation):
        voice_sections, filtered_sections = [], []
        for s in segmentation:
            kind, start, stop = s
            if kind == 'female' or kind == 'male':
                voice_sections.append((start, stop))
            else:
                filtered_sections.append((start, stop))
        return voice_sections, filtered_sections

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        output_dir = self.config['output_dir']
        filtered_dir = self.config['filtered_dir']
        create_dirs_if_not_exist(output_dir, filtered_dir)

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload

        p_df = pd.DataFrame()
        processed_path = f'{self.get_name()}_processed_path'
        metadata['paths_column'] = processed_path
        metadata['all_paths_columns'].append(processed_path)
        for f in paths_list:
            try:
                start = time.time()
                segmentation = self.model(f)
                v_segments, f_segments = INAVoiceSeparator.get_voice_segments(segmentation)
                for i, segment in enumerate(v_segments):
                    output_path = cut_segment(f, output_dir=output_dir, segment=segment, segment_id=i)
                    f_df = pd.DataFrame.from_dict({processed_path: [output_path],
                                                   f'{self.get_name()}_segment_start': [segment[0]],
                                                   f'{self.get_name()}_segment_stop': [segment[1]],
                                                   input_column: [f]})
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
                end = time.time()
                self.logger.info(f'Extracted {len(v_segments)} from {f} in {end - start} seconds')

            except AssertionError as err:
                self.logger.error(f"Error reading {f}.\n{err}")

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
