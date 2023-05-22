import time

from yaml import YAMLObject
import torchaudio
import torch
import pandas as pd
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import get_null_wav_path, cached_download
import torch.optim as optim
import vanpy_speaker_embedding

class VanpySpeakerEmbedding(PipelineComponent):
    model = None

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='vanpy_speaker_embedding',
                         yaml_config=yaml_config)
        self.pretrained_models_dir = self.config['pretrained_models_dir']

    def load_model(self):
        # if self.config['model']:
        #     mdl = self.config['model']
        model_path = cached_download('https://drive.google.com/uc?id=1CW5zuEzIIjj2kf2AtG-Xx61couMcGgLi',
                                     f'{self.pretrained_models_dir}/speaker_embedding_mozcv_step_200.pkl')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = vanpy_speaker_embedding.ConformerModel(n_mels=128, n_time_steps=100, n_conformer_blocks=2, d_model=128, nhead=4,
                                   n_embeddings=192).to(self.device)
        optimizer = optim.Adam(model.parameters())
        self.model, _, _, _, _ = vanpy_speaker_embedding.load_checkpoint(model, optimizer, model_path)
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        df = df.reset_index().drop(['index'], axis=1, errors='ignore')
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        self.config['records_count'] = len(paths_list)

        file_performance_column_name = ''
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_features'
            metadata['meta_columns'].extend([file_performance_column_name])
            if file_performance_column_name in df.columns:
                df = df.drop([file_performance_column_name], axis=1)
            df.insert(0, file_performance_column_name, None)

        df, feature_columns = self.create_and_get_feature_columns(df)

        for j, f in enumerate(paths_list):
            try:
                t_start_feature_extraction = time.time()
                embedding = vanpy_speaker_embedding.extract_embeddings([f], self.model, self.device, audio_dir=None)
                f_df = pd.DataFrame(embedding.to('cpu').numpy().ravel()).T
                f_df[input_column] = f
                t_end_feature_extraction = time.time()
                if self.config['performance_measurement']:
                    f_df[file_performance_column_name] = t_end_feature_extraction - t_start_feature_extraction
                for c in f_df.columns:
                    df.at[j, f'{c}_{self.get_name()}'] = f_df.iloc[0, f_df.columns.get_loc(c)]
                self.latent_info_log(f'done with {f}, {j + 1}/{len(paths_list)}', iteration=j)
            except (TypeError, RuntimeError) as e:
                self.logger.error(f'An error occurred in {f}, {j + 1}/{len(paths_list)}: {e}')
            self.save_intermediate_payload(j, ComponentPayload(metadata=metadata, df=df))

        metadata['feature_columns'].extend(feature_columns)
        return ComponentPayload(metadata=metadata, df=df)

    def create_and_get_feature_columns(self, df: pd.DataFrame):
        feature_columns = []
        embedding = vanpy_speaker_embedding.extract_embeddings([get_null_wav_path()], self.model, self.device, audio_dir=None)
        f_df = pd.DataFrame(embedding.to('cpu').numpy().ravel()).T
        for c in f_df.columns[::-1]:
            c = f'{c}_{self.get_name()}'
            feature_columns.append(c)
            if c not in df.columns:
                df.insert(0, c, None)
        return df, feature_columns
