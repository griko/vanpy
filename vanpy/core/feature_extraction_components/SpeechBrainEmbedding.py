import time

from yaml import YAMLObject
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import pandas as pd
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PiplineComponent import PipelineComponent


class SpeechBrainEmbedding(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='speechbrain_embedding',
                         yaml_config=yaml_config)

    def load_model(self):
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()

        file_performance_column_name = ''
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_features'
            metadata['meta_columns'].extend([file_performance_column_name])
        p_df = pd.DataFrame()
        for f in paths_list:
            try:
                t_start_feature_extraction = time.time()
                signal, fs = torchaudio.load(f)
                embedding = self.model.encode_batch(signal)
                f_df = pd.DataFrame(embedding.to('cpu').numpy().ravel()).T
                f_df[input_column] = f
                t_end_feature_extraction = time.time()
                if self.config['performance_measurement']:
                    f_df[file_performance_column_name] = t_end_feature_extraction - t_start_feature_extraction
                p_df = pd.concat([p_df, f_df], ignore_index=True)
                self.logger.info(f'done with {f}')
            except (TypeError, RuntimeError) as e:
                self.logger.error(f'An error occurred in {f}: {e}')

        feature_columns = [str(x) for x in range(512)]
        feature_columns.remove(input_column)
        metadata['feature_columns'].extend(feature_columns)
        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
