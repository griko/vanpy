import time
from yaml import YAMLObject
from pyannote.audio import Inference
from pyannote.audio import Model
import numpy as np
import pandas as pd
import torch
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import get_null_wav_path


class PyannoteEmbedding(PipelineComponent):
    """
    A feature extraction component that uses Pyannote models to generate speaker embeddings.

    :ivar model: The loaded Pyannote inference model instance.
    """
    model = None

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the Pyannote embedding component.

        :param yaml_config: Configuration parameters for the component.
        :raises KeyError: If huggingface_ACCESS_TOKEN is not provided in the config.
        """
        super().__init__(component_type='feature_extraction', component_name='pyannote_embedding',
                         yaml_config=yaml_config)
        self.ACCESS_TOKEN = self.config.get('huggingface_ACCESS_TOKEN', None)
        if self.ACCESS_TOKEN is None:
            raise KeyError(f'You need to pass huggingface_ACCESS_TOKEN to use {self.component_name} model')

    def load_model(self):
        """
        Load and initialize the Pyannote embedding model.
        Automatically selects GPU if available, otherwise uses CPU.
        """
        model = Model.from_pretrained("pyannote/embedding",  # pyannote%2Fembedding
                                      use_auth_token=self.ACCESS_TOKEN)
        if torch.cuda.is_available():
            self.model = Inference(model,
                                   window="sliding",
                                   duration=self.config['sliding_window_duration'],
                                   step=self.config['sliding_window_step'],
                                   device="cuda")
        else:
            self.model = Inference(model,
                                   window="sliding",
                                   duration=self.config['sliding_window_duration'],
                                   step=self.config['sliding_window_step'])
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

    def process_item(self, f, input_column):
        """
        Process a single audio file to extract embeddings.

        :param f: Path to the audio file.
        :param input_column: Name of the column containing file paths.
        :return: DataFrame containing the extracted embeddings.
        """
        embedding = self.model(f)
        f_df = pd.DataFrame(np.mean(embedding, axis=0)).T
        f_df[input_column] = f
        f_df.rename(columns={i: c for i, c in enumerate(self.get_feature_columns())}, inplace=True)
        return f_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process a batch of audio files to extract embeddings.

        :param input_payload: Input payload containing audio file paths and metadata.
        :return: Output payload containing the extracted embeddings.
        """
        if not self.model:
            self.load_model()

        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        paths_list = df[input_column].dropna().tolist()

        metadata = self.add_performance_column_to_metadata(metadata)

        p_df = self.process_with_progress(paths_list, metadata, input_column)

        df = pd.merge(left=df, right=p_df, how='left', on=input_column)
        return ComponentPayload(metadata=metadata, df=df)

    def get_feature_columns(self):
        """
        Generate the list of feature column names.

        :return: List of column names for the extracted features.
        """
        feature_columns = []
        embedding = self.model(get_null_wav_path())
        f_df = pd.DataFrame(np.mean(embedding, axis=0)).T
        for c in f_df.columns:
            c = f'{c}_{self.get_name()}'
            feature_columns.append(c)
        return feature_columns
