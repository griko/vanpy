import os
from yaml import YAMLObject
from audio_pipeline.core.PiplineComponent import PipelineComponent, ComponentPayload
from speechbrain.pretrained.interfaces import foreign_class


class IEMOCAPEmotionClassifier(PipelineComponent):
    model = None
    classification_column_name: str = ''
    pretrained_models_dir: str = ''
    verbal_labels: bool = False

    def __init__(self, yaml_config: YAMLObject):
        super().__init__(component_type='segment_classifier', component_name='speech_brain_iemocap_emotion',
                         yaml_config=yaml_config)
        self.verbal_labels = self.config['verbal_labels']
        self.classification_column_name = self.config['classification_column_name']
        self.pretrained_models_dir = self.config['pretrained_models_dir']

    def load_model(self):
        self.logger.info("Loading emotion classification model, trained on IEMOCAP "
                         "dataset with Speech Brain")
        self.model = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                                   pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier",
                                   savedir=self.pretrained_models_dir)

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        if not self.model:
            self.load_model()

        payload_metadata, payload_df = input_payload.unpack()
        input_column = payload_metadata['paths_column']
        paths_list = payload_df[input_column].tolist()

        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            return input_payload
        
        emotion_prediction = []
        for f in paths_list:
            out_prob, score, index, text_lab = self.model.classify_file(f)
            emotion_prediction.append(text_lab[0] if self.verbal_labels else index)

        payload_df[self.classification_column_name] = emotion_prediction
        payload_metadata['classification_columns'].extend([self.classification_column_name])

        IEMOCAPEmotionClassifier.cleanup_softlinks()
        return ComponentPayload(metadata=payload_metadata, df=payload_df)

    @classmethod
    def cleanup_softlinks(cls):
        for link in os.listdir():
            if '.wav' in link and os.path.islink(link):
                os.unlink(link)
