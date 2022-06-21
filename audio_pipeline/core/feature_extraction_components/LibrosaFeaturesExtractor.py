import librosa
from yaml import YAMLObject
import numpy as np
import pandas as pd

from audio_pipeline.core.ComponentPayload import ComponentPayload
from audio_pipeline.core.PiplineComponent import PipelineComponent


class LibrosaFeaturesExtractor(PipelineComponent):
    features: list[str] = None
    sampling_rate: int

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='feature_extraction', component_name='librosa_features_extractor',
                         yaml_config=yaml_config)
        self.sampling_rate = self.config['sampling_rate']
        self.features = self.config['features']
        self.n_mfcc = self.config['n_mfcc']

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].tolist()

        p_df = pd.DataFrame()
        for f in paths_list:
            try:
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

                p_df = pd.concat([p_df, f_df], ignore_index=True)
                self.logger.info(f'done with {f}')
            except RuntimeError as e:
                self.logger.error(f'An error occurred in {f}: {e}')

        feature_columns = p_df.columns.tolist()
        feature_columns.remove(input_column)
        metadata['feature_columns'].extend(feature_columns)
        df = pd.merge(left=df, right=p_df, how='outer', left_on=input_column, right_on=input_column)
        return ComponentPayload(metadata=metadata, df=df)
