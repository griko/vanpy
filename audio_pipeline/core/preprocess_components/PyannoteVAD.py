from yaml import YAMLObject

from audio_pipeline.core.ComponentPayload import ComponentPayload
from audio_pipeline.core.PiplineComponent import PipelineComponent
from audio_pipeline.utils.utils import create_dirs_if_not_exist, cut_segment
from pyannote.audio.pipelines import VoiceActivityDetection
import pandas as pd
import time


class PyannoteVAD(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='preprocessing', component_name='pyannote_vad',
                         yaml_config=yaml_config)
        self.params = {   # onset/offset activation thresholds
                          "onset": self.config['onset'], "offset": self.config['offset'],
                          # remove speech regions shorter than that many seconds.
                          "min_duration_on": self.config['min_duration_on'],
                          # fill non-speech regions shorter than that many seconds.
                          "min_duration_off": self.config['min_duration_off']
                        }

    def load_model(self):
        self.model = VoiceActivityDetection(segmentation="pyannote/segmentation")
        self.model.instantiate(self.params)

    @staticmethod
    def get_voice_segments(segmentation):
        sections = []
        for i, v in enumerate(segmentation.itersegments()):
            start, stop = v
            sections.append((start, stop))
        return sections

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
        segment_start_column_name = f'{self.get_name()}_segment_start'
        segment_stop_column_name = f'{self.get_name()}_segment_stop'
        metadata['meta_columns'].extend([segment_start_column_name, segment_stop_column_name])
        for f in paths_list:
            try:
                start = time.time()
                v_segments = PyannoteVAD.get_voice_segments(self.model(f))
                for i, segment in enumerate(v_segments):
                    output_path = cut_segment(f, output_dir=output_dir, segment=segment, segment_id=i)
                    f_df = pd.DataFrame.from_dict({processed_path: [output_path],
                                                   segment_start_column_name: [segment[0]],
                                                   segment_stop_column_name: [segment[1]],
                                                   input_column: [f]})
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
                end = time.time()
                self.logger.info(f'Extracted {len(v_segments)} from {f} in {end - start} seconds')
            except RuntimeError as err:
                self.logger.error(f"Could not create VAD pipline for {f} with pyannote.\n{err}")

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
