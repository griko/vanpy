import os
import pandas as pd
from yaml import YAMLObject

from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent


class IEMOCAPEmotionClassifier(PipelineComponent):
    model = None
    classification_column_name: str = ''
    verbal_labels: bool = True

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='speech_brain_iemocap_emotion',
                         yaml_config=yaml_config)
        self.verbal_labels = self.config.get('verbal_labels', True)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')

    def load_model(self):
        from speechbrain.pretrained.interfaces import foreign_class
        self.logger.info("Loading emotion classification model, trained on IEMOCAP "
                         "dataset with Speech Brain")
        self.model = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                                   pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier",
                                   savedir=self.pretrained_models_dir)

    def process_item(self, f, p_df, input_column, classification_column_name):
        try:
            out_prob, score, index, text_lab = self.model.classify_file(f)
            emotion_prediction = text_lab[0] if self.verbal_labels else index
        except (RuntimeError, TypeError) as e:
            emotion_prediction = None
            self.logger.error(f"An error occurred in {f}: {e}")

        f_df = pd.DataFrame({input_column: [f], classification_column_name: [emotion_prediction]})
        p_df = pd.concat([p_df, f_df], ignore_index=True)
        return p_df

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].tolist()

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload

        p_df = pd.DataFrame()
        payload_metadata['classification_columns'].append(self.classification_column_name)

        p_df = self.process_with_progress(
            paths_list,
            payload_metadata,
            self.process_item,
            p_df,
            input_column,
            self.classification_column_name
        )

        payload_df = pd.merge(
            left=payload_df,
            right=p_df,
            how='outer',
            left_on=input_column,
            right_on=input_column
        )

        return ComponentPayload(metadata=payload_metadata, df=payload_df)

    @staticmethod
    def cleanup_softlinks():
        for link in os.listdir():
            if '.wav' in link and os.path.islink(link):
                os.unlink(link)
