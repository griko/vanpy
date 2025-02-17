import pandas as pd
from yaml import YAMLObject
import numpy as np
import requests
import tarfile
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import create_dirs_if_not_exist


class YamnetClassifier(PipelineComponent):
    """
    Audio classification component using the YAMNet model for sound event detection.
    
    :ivar model: Loaded YAMNet model instance.
    :ivar classification_column_name: Output column name for sound classifications.
    :ivar class_names: List of available sound classes.
    :ivar threshold: Confidence threshold for class prediction.
    """
    model = None
    classification_column_name: str = ''
    class_names = None
    threshold = 0.

    def __init__(self, yaml_config: YAMLObject):
        """
        Initialize the YAMNet classifier.

        :param yaml_config: Configuration parameters for the classifier.
        """
        super().__init__(component_type='segment_classifier', component_name='yamnet_classifier',
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name', f'{self.component_name}_classification')
        self.top_k = self.config.get('top_k', 1)
        self.threshold = self.config.get('threshold', 0.)

    @staticmethod
    # Find the name of the class with the top score when mean-aggregated across frames.
    def class_names_from_csv(class_map_csv_text):
        """
        Extract class names from the YAMNet class mapping file.

        :param class_map_csv_text: Path to the class mapping CSV file.
        :return: List of class names for sound classification.
        """
        import tensorflow as tf
        import csv
        """Returns list of class names corresponding to score vector."""
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])

        return class_names

    @staticmethod
    def ensure_sample_rate(original_sample_rate, waveform,
                           desired_sample_rate=16000):
        """
        Resample audio waveform to the desired sample rate.

        :param original_sample_rate: Current sampling rate of the waveform.
        :param waveform: Audio waveform data.
        :param desired_sample_rate: Target sampling rate (default: 16000).
        :return: Tuple of (new_sample_rate, resampled_waveform).
        """
        import scipy
        """Resample waveform if required."""
        if original_sample_rate != desired_sample_rate:
            desired_length = int(round(float(len(waveform)) /
                                       original_sample_rate * desired_sample_rate))
            waveform = scipy.signal.resample(waveform, desired_length)
        return desired_sample_rate, waveform

    def load_model(self):
        """
        Load the YAMNet model and class mappings.
        
        Downloads and extracts the model if not present in the pretrained models directory.
        """
        import os.path
        from os import path
        import tensorflow_hub as hub

        self.logger.info("Loading Yamnet v1 sound classification model, 512 labels, trained on AudioSet-YouTube corpus")
        model_path = self.pretrained_models_dir
        if not path.exists(model_path):
            create_dirs_if_not_exist(model_path)
            r = requests.get('https://storage.googleapis.com/tfhub-modules/google/yamnet/1.tar.gz', allow_redirects=True)
            with open(f'{model_path}/1.tar.gz', 'wb') as f:
                f.write(r.content)
            with tarfile.open(f'{model_path}/1.tar.gz') as f:
                f.extractall(model_path)
            os.remove(f'{model_path}/1.tar.gz')
        self.model = hub.load(model_path)
        class_map_path = self.model.class_map_path().numpy()
        self.class_names = YamnetClassifier.class_names_from_csv(class_map_path)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Process audio files to predict sound event classes.

        :param input_payload: Input payload containing audio file paths and metadata.
        :return: Output payload containing sound classifications.
        """
        import tensorflow as tf
        from scipy.io import wavfile

        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].dropna().tolist()

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload

        class_prediction = []

        p_df = pd.DataFrame()
        for j, f in enumerate(paths_list):
            try:
                sample_rate, wav_data = wavfile.read(f)
                waveform = wav_data / tf.int16.max
                scores, embeddings, spectrogram = self.model(waveform)
                mean_scores = np.mean(scores.numpy(), axis=0)
                top_class_indices = np.argsort(mean_scores)[::-1][:self.top_k]
                top_class_indices_refined = []
                for idx in top_class_indices:
                    if mean_scores[idx] >= self.threshold:
                        top_class_indices_refined.append(idx)
                inferred_class = '; '.join([self.class_names[x] for x in top_class_indices_refined])  # self.class_names[scores_np.mean(axis=0).argmax()]
                f_df = pd.DataFrame({input_column: [f], self.classification_column_name: [inferred_class]})
                p_df = pd.concat([p_df, f_df], ignore_index=True)
                # class_prediction.append(inferred_class)
            except (FileNotFoundError, RuntimeError, TypeError) as e:
                # class_prediction.append(None)
                self.logger.error(f"An error occurred in {f}, {j + 1}/{len(paths_list)}: {e}")

        payload_df = pd.merge(left=payload_df, right=p_df, how='left', on=input_column)
        # payload_df[self.classification_column_name] = class_prediction
        payload_metadata['classification_columns'].extend([self.classification_column_name])
        return ComponentPayload(metadata=payload_metadata, df=payload_df)
    #
    # @staticmethod
    # def cleanup_softlinks():
    #     for link in os.listdir():
    #         if '.wav' in link and os.path.islink(link):
    #             os.unlink(link)
