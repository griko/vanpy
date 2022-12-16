import time

from yaml import YAMLObject
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import pandas as pd
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PiplineComponent import PipelineComponent
from vanpy.utils.utils import get_null_wav_path


class SpeechBrainEmbedding(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='speechbrain_embedding',
                         yaml_config=yaml_config)

    def load_model(self):
        mdl = self.config['model'] if self.config['model'] else 'spkrec-ecapa-voxceleb'
        self.model = EncoderClassifier.from_hparams(source=f"speechbrain/{mdl}", savedir=f"pretrained_models/{mdl}")

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        df = df.reset_index().drop(['index'], axis=1, errors='ignore')
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()

        file_performance_column_name = ''
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_features'
            metadata['meta_columns'].extend([file_performance_column_name])
            if file_performance_column_name in df.columns:
                df = df.drop([file_performance_column_name], axis=1)
            df.insert(0, file_performance_column_name, None)

        # replace the feature columns
        signal, fs = torchaudio.load(get_null_wav_path())
        embedding = self.model.encode_batch(signal)
        f_df = pd.DataFrame(embedding.to(self.config['device']).numpy().ravel()).T
        for c in f_df.columns[::-1]:
            c = f'{c}_{self.get_name()}'
            if c in df.columns:
                df = df.drop([c], axis=1)
            df.insert(0, c, None)

        for j, f in enumerate(paths_list):
            try:
                t_start_feature_extraction = time.time()
                signal, fs = torchaudio.load(f)
                embedding = self.model.encode_batch(signal)
                f_df = pd.DataFrame(embedding.to(self.config['device']).numpy().ravel()).T
                f_df[input_column] = f
                t_end_feature_extraction = time.time()
                if self.config['performance_measurement']:
                    f_df[file_performance_column_name] = t_end_feature_extraction - t_start_feature_extraction
                for c in f_df.columns:
                    df.at[j, f'{c}_{self.get_name()}'] = f_df.iloc[0, f_df.columns.get_loc(c)]
                self.logger.info(f'done with {f}, {j}/{len(paths_list)}')
            except (TypeError, RuntimeError) as e:
                self.logger.error(f'An error occurred in {f}, {j}/{len(paths_list)}: {e}')

        feature_columns = [str(x) for x in range(512)]
        metadata['feature_columns'].extend(feature_columns)
        return ComponentPayload(metadata=metadata, df=df)
