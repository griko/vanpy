import librosa
from librosa.util.exceptions import ParameterError
from yaml import YAMLObject
import numpy as np
import pandas as pd
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from typing import List
import logging


class LibrosaFeaturesExtractor(PipelineComponent):
    features: List[str] = None
    sampling_rate: int

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='librosa_features_extractor',
                         yaml_config=yaml_config)
        self.sampling_rate = self.config.get('sampling_rate', 16000)
        self.features = self.config.get('features', ['mfcc'])
        self.n_mfcc = self.config.get('n_mfcc', 13)

        # Disable numba DEBUG logs
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

    def process_item(self, f, p_df, input_column):
        try:
            y, sr = librosa.load(f, sr=self.sampling_rate)
            f_df = pd.DataFrame()

            if 'mfcc' in self.features:
                mfcc = librosa.feature.mfcc(y=y, sr=self.sampling_rate, n_mfcc=self.n_mfcc)
                mean_mfcc = np.mean(mfcc, axis=1)
                for i, val in enumerate(mean_mfcc):
                    f_df[f'mfcc_{i}'] = [val]

                if 'delta_mfcc' in self.features:
                    mean_delta_mfcc = np.mean(librosa.feature.delta(mfcc, mode='nearest'), axis=1)
                    for i, val in enumerate(mean_delta_mfcc):
                        f_df[f'd_mfcc_{i}'] = [val]

            if 'zero_crossing_rate' in self.features:
                f_df['zero_crossing_rate'] = [np.count_nonzero(np.array(librosa.zero_crossings(y, pad=False))) / len(y)]

            if 'spectral_centroid' in self.features:
                f_df['spectral_centroid'] = [np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sampling_rate))]

            if 'spectral_bandwidth' in self.features:
                f_df['spectral_bandwidth'] = [np.mean(librosa.feature.spectral_bandwidth(y=y, sr=self.sampling_rate))]

            if 'spectral_contrast' in self.features:
                f_df['spectral_contrast'] = [np.mean(librosa.feature.spectral_contrast(y=y, sr=self.sampling_rate))]

            if 'spectral_flatness' in self.features:
                f_df['spectral_flatness'] = [np.mean(librosa.feature.spectral_flatness(y=y))]

            if 'f0' in self.features:
                f0, _voiced_flag, _voiced_probs = librosa.pyin(y=y, sr=self.sampling_rate,
                                                               fmin=librosa.note_to_hz('C2'),
                                                               fmax=librosa.note_to_hz('C7'))
                f_df['f0'] = [np.mean(f0)]

            if 'tonnetz' in self.features:
                f_df['tonnetz'] = [np.mean(librosa.feature.tonnetz(y=y, sr=self.sampling_rate))]

            f_df[input_column] = f
            p_df = pd.concat([p_df, f_df], ignore_index=True)

        except (RuntimeError, TypeError, ParameterError) as e:
            self.logger.error(f'An error occurred processing {f}: {e}')

        return p_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()
        self.config['records_count'] = len(paths_list)

        p_df = pd.DataFrame()
        metadata = self.add_performance_column_to_metadata(metadata)

        p_df = self.process_with_progress(paths_list, metadata, self.process_item, p_df, input_column)

        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)

        # Add feature columns to metadata
        feature_columns = self.get_feature_columns()
        metadata['feature_columns'].extend(feature_columns)

        return ComponentPayload(metadata=metadata, df=df)

    def get_feature_columns(self):
        feature_columns = []
        # Add MFCC columns
        if 'mfcc' in self.features:
            feature_columns.extend([f'mfcc_{i}' for i in range(self.n_mfcc)])
            if 'delta_mfcc' in self.features:
                feature_columns.extend([f'd_mfcc_{i}' for i in range(self.n_mfcc)])

        # Add other feature columns
        other_features = ['zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth',
                          'spectral_contrast', 'spectral_flatness', 'f0', 'tonnetz']
        feature_columns.extend([feat for feat in other_features if feat in self.features])

        return feature_columns
