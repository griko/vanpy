import time

import librosa
from librosa.util.exceptions import ParameterError
from yaml import YAMLObject
import numpy as np
import pandas as pd
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from typing import List
from tqdm.auto import tqdm


class LibrosaFeaturesExtractor(PipelineComponent):
    features: List[str] = None
    sampling_rate: int

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='librosa_features_extractor',
                         yaml_config=yaml_config)
        self.sampling_rate = self.config.get('sampling_rate', 16000)
        self.features = self.config.get('features', ['mfcc'])
        self.n_mfcc = self.config.get('n_mfcc', 13)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        metadata, df = input_payload.unpack()
        df = df.reset_index().drop(['index'], axis=1, errors='ignore')
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        self.config['records_count'] = len(paths_list)

        file_performance_column_name = ''
        if self.config['performance_measurement']:
            file_performance_column_name = f'perf_{self.get_name()}_get_features'
            metadata['meta_columns'].extend([file_performance_column_name])

        df, feature_columns = self.create_and_get_feature_columns(df)

        for j, f in enumerate(tqdm(paths_list)):
            try:
                t_start_feature_extraction = time.time()
                y, sr = librosa.load(f, sr=self.sampling_rate)
                f_df = pd.DataFrame([{input_column: f}])
                if 'mfcc' in self.features:
                    mfcc = librosa.feature.mfcc(y=y, sr=self.sampling_rate, n_mfcc=self.n_mfcc)
                    mean_mfcc = np.mean(mfcc, axis=1)
                    mfcc_df = pd.DataFrame(mean_mfcc, index=[f'mfcc_{i}' for i in range(self.n_mfcc)]).T
                    f_df = pd.merge(left=f_df, right=mfcc_df, left_index=True, right_index=True)
                    if 'delta_mfcc' in self.features:
                        mean_delta_mfcc = np.mean(librosa.feature.delta(mfcc), axis=1)
                        d_mfcc_df = pd.DataFrame(mean_delta_mfcc, index=[f'd_mfcc_{i}' for i in range(self.n_mfcc)]).T
                        f_df = pd.merge(left=f_df, right=d_mfcc_df, left_index=True, right_index=True)
                if 'zero_crossing_rate' in self.features:
                    zero_crossing_rate = np.count_nonzero(np.array(librosa.zero_crossings(y, pad=False)))/len(y)
                    f_df['zero_crossing_rate'] = zero_crossing_rate
                if 'spectral_centroid' in self.features:
                    f_df['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sampling_rate))
                if 'spectral_bandwidth' in self.features:
                    f_df['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=self.sampling_rate))
                if 'spectral_contrast' in self.features:
                    f_df['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=self.sampling_rate))
                if 'spectral_flatness' in self.features:
                    f_df['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))
                if 'f0' in self.features:
                    f0, _voiced_flag, _voiced_probs = librosa.pyin(y=y, sr=self.sampling_rate,
                                                                 fmin=librosa.note_to_hz('C2'),
                                                                 fmax=librosa.note_to_hz('C7'))
                    f_df['f0'] = np.mean(f0)
                if 'tonnetz' in self.features:
                    f_df['tonnetz'] = np.mean(librosa.feature.tonnetz(y=y, sr=self.sampling_rate))

                t_end_feature_extraction = time.time()
                if self.config['performance_measurement']:
                    f_df[file_performance_column_name] = t_end_feature_extraction - t_start_feature_extraction

                for c in f_df.columns:
                    df.at[j, c] = f_df.iloc[0, f_df.columns.get_loc(c)]
                self.latent_info_log(f'done with {f}, {j + 1}/{len(paths_list)}', iteration=j)
            except (RuntimeError, TypeError, ParameterError) as e:
                self.logger.error(f'An error occurred in {f}, {j + 1}/{len(paths_list)}: {e}')
            self.save_intermediate_payload(j, ComponentPayload(metadata=metadata, df=df))

        metadata['feature_columns'].extend(feature_columns)
        return ComponentPayload(metadata=metadata, df=df)

    def create_and_get_feature_columns(self, df: pd.DataFrame):
        feature_columns = []
        cols = ['zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
                'spectral_flatness', 'f0', 'tonnetz']
        for col in cols:
            if self.config['features'] and col in self.config['features']:
                feature_columns.append(col)
        if self.config['features'] and 'mfcc' in self.config['features']:
            feature_columns.extend([f'mfcc_{i}' for i in range(self.n_mfcc)])
        if self.config['features'] and 'delta_mfcc' in self.config['features']:
            feature_columns.extend([f'd_mfcc_{i}' for i in range(self.n_mfcc)])

        for c in feature_columns[::-1]:
            if c not in df.columns:
                df.insert(0, c, None)

        return df, feature_columns
