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
    """
    A feature extraction component that uses librosa to extract audio features.

    :ivar features: List of feature types to extract.
    :ivar sampling_rate: Target sampling rate for audio processing.
    """
    features: List[str] = None
    sampling_rate: int

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the librosa feature extractor.

        :param yaml_config: Configuration parameters for the component.
        """
        super().__init__(component_type='feature_extraction', component_name='librosa_features_extractor',
                         yaml_config=yaml_config)
        self.sampling_rate = self.config.get('sampling_rate', 16000)
        self.features = self.config.get('features', ['mfcc'])
        self.n_mfcc = self.config.get('n_mfcc', 13)

        self.feature_columns = self.get_feature_columns()

        # Disable numba DEBUG logs
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

    def process_item(self, f, input_column):
        """
        Process a single audio file to extract librosa features.

        :param f: Path to the audio file.
        :param input_column: Name of the column containing file paths.
        :return: DataFrame containing the extracted features.
        """
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

        except (FileNotFoundError, RuntimeError, TypeError, ParameterError) as e:
            self.logger.error(f'An error occurred processing {f}: {e}')

        return f_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process a batch of audio files to extract librosa features.

        :param input_payload: Input payload containing audio file paths and metadata.
        :return: Output payload containing the extracted features.
        """
        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].dropna().tolist()

        metadata = self.add_performance_column_to_metadata(metadata)

        p_df = self.process_with_progress(paths_list, metadata, input_column)

        df = pd.merge(left=df, right=p_df, how='left', on=input_column)

        # Add feature columns to metadata
        feature_columns = self.feature_columns
        metadata['feature_columns'].extend(feature_columns)

        return ComponentPayload(metadata=metadata, df=df)

    def get_feature_columns(self):
        """
        Generate the list of feature column names based on configured features.

        :return: List of column names for the extracted features.
        """
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
