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
    """
    A feature extraction component that uses SpeechBrain models to generate embeddings from audio files.

    :ivar model: The loaded SpeechBrain encoder model instance.
    :ivar feature_columns: List of column names for the extracted features.
    """
    model = None

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the SpeechBrain embedding component.

        :param yaml_config: Configuration parameters for the component.
        """
        super().__init__(component_type='feature_extraction', component_name='speechbrain_embedding',
                         yaml_config=yaml_config)
        self.feature_columns = None

    def load_model(self):
        """
        Load and initialize the SpeechBrain encoder model.
        Automatically selects GPU if available, otherwise uses CPU.
        """
        mdl = self.config.get('model', 'spkrec-ecapa-voxceleb')
        if torch.cuda.is_available():
            self.model = EncoderClassifier.from_hparams(source=f"speechbrain/{mdl}", savedir=f"pretrained_models/{mdl}",
                                                        run_opts={"device": "cuda"})
        else:
            self.model = EncoderClassifier.from_hparams(source=f"speechbrain/{mdl}", savedir=f"pretrained_models/{mdl}")
        self.logger.info(f'Loaded model to {"GPU" if torch.cuda.is_available() else "CPU"}')

    def process_item(self, f, input_column):
        """
        Process a single audio file to extract embeddings.

        :param f: Path to the audio file.
        :param input_column: Name of the column containing file paths.
        :return: DataFrame containing the extracted embeddings.
        """
        signal, fs = torchaudio.load(f)
        embedding = self.model.encode_batch(signal)
        f_df = pd.DataFrame(embedding.to('cpu').numpy().ravel()).T
        f_df.columns = [c for c in self.feature_columns]
        f_df[input_column] = f
        return f_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process a batch of audio files to extract embeddings.

        :param input_payload: Input payload containing audio file paths and metadata.
        :return: Output payload containing the extracted embeddings.
        """
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
        """
        Generate the list of feature column names.

        :return: List of column names for the extracted features.
        """
        feature_columns = []
        signal, fs = torchaudio.load(get_null_wav_path())
        embedding = self.model.encode_batch(signal)
        f_df = pd.DataFrame(embedding.to('cpu').numpy().ravel()).T
        for c in f_df.columns:
            c = f'{c}_{self.get_name()}'
            feature_columns.append(c)

        return feature_columns
