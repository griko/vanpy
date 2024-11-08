import time

from yaml import YAMLObject
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
import pandas as pd
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import get_null_wav_path


class SpeechBrainEmbedding(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='speechbrain_embedding',
                         yaml_config=yaml_config)
        self.feature_columns = None

    def load_model(self):
        mdl = 'spkrec-ecapa-voxceleb'  # default model
        if self.config['model']:
            mdl = self.config['model']
        if torch.cuda.is_available():
            self.model = EncoderClassifier.from_hparams(source=f"speechbrain/{mdl}", savedir=f"pretrained_models/{mdl}",
                                                        run_opts={"device": "cuda"})
        else:
            self.model = EncoderClassifier.from_hparams(source=f"speechbrain/{mdl}", savedir=f"pretrained_models/{mdl}")
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

    def process_item(self, f, input_column):
        signal, fs = torchaudio.load(f)
        embedding = self.model.encode_batch(signal)
        f_df = pd.DataFrame(embedding.to('cpu').numpy().ravel()).T
        f_df.columns = [c for c in self.feature_columns]
        f_df[input_column] = f
        return f_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        self.feature_columns = self.get_feature_columns()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].dropna().tolist()

        metadata = self.add_performance_column_to_metadata(metadata)

        p_df = self.process_with_progress(paths_list, metadata, input_column)

        df = pd.merge(left=df, right=p_df, how='left', on=input_column)
        return ComponentPayload(metadata=metadata, df=df)

    def get_feature_columns(self):
        feature_columns = []
        signal, fs = torchaudio.load(get_null_wav_path())
        embedding = self.model.encode_batch(signal)
        f_df = pd.DataFrame(embedding.to('cpu').numpy().ravel()).T
        for c in f_df.columns:
            c = f'{c}_{self.get_name()}'
            feature_columns.append(c)

        return feature_columns
