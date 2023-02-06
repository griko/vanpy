from typing import List
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline


class ClassificationPipeline(BasePipeline):
    components_mapper = {
        'common_voices_gender': None,
        'common_voices_age': None,
        'speech_brain_iemocap_emotion': None,
        'wav2vec2stt': None,
        'openai_whisper_stt': None
    }

    def __init__(self, components: List[str], config: YAMLObject):
        for component in components:
            if component == 'common_voices_gender':
                from vanpy.core.segment_classification_components.CVGenderClassifier import CVGenderClassifier
                self.components_mapper[component] = CVGenderClassifier
            elif component == 'common_voices_age':
                from vanpy.core.segment_classification_components.CVAgeClassifier import CVAgeClassifier
                self.components_mapper[component] = CVAgeClassifier
            elif component == 'speech_brain_iemocap_emotion':
                from vanpy.core.segment_classification_components.IEMOCAPEmotionClassifier import IEMOCAPEmotionClassifier
                self.components_mapper[component] = IEMOCAPEmotionClassifier
            elif component == 'wav2vec2stt':
                from vanpy.core.segment_classification_components.Wav2Vec2STT import Wav2Vec2STT
                self.components_mapper[component] = Wav2Vec2STT
            elif component == 'openai_whisper_stt':
                from vanpy.core.segment_classification_components.WhisperSTT import WhisperSTT
                self.components_mapper[component] = WhisperSTT
        super().__init__(components, config)
        self.logger.info(f'Created Classification Pipeline with {len(self.components)} components')
