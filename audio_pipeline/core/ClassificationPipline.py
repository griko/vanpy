from typing import List
from yaml import YAMLObject
from audio_pipeline.core.BasePipline import BasePipeline
from audio_pipeline.core.segment_classification_components.CVAgeClassifier import CVAgeClassifier
from audio_pipeline.core.segment_classification_components.CVGenderClassifier import CVGenderClassifier
from audio_pipeline.core.segment_classification_components.IEMOCAPEmotionClassifier import IEMOCAPEmotionClassifier
from audio_pipeline.core.segment_classification_components.Wav2Vec2STT import Wav2Vec2STT


class ClassificationPipeline(BasePipeline):
    components_mapper = {
        'common_voices_gender': CVGenderClassifier,
        'common_voices_age': CVAgeClassifier,
        'speech_brain_iemocap_emotion': IEMOCAPEmotionClassifier,
        'wav2vec2stt': Wav2Vec2STT
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Classification Pipeline with {len(self.components)} components')
