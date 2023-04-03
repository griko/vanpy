from typing import List
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline


class ClassificationPipeline(BasePipeline):
    components_mapper = {
        'vanpy_voxceleb_gender': None,
        'vanpy_voxceleb_age': None,
        'vanpy_voxceleb_height': None,
        # 'common_voices_gender': None,
        # 'common_voices_age': None,
        'speech_brain_iemocap_emotion': None,
        'vanpy_ravdess_emotion': None,
        'wav2vec2adv': None,
        'wav2vec2stt': None,
        'openai_whisper_stt': None,
        'cosine_distance_diarization': None,
        'yamnet_classifier': None
    }

    def __init__(self, components: List[str], config: YAMLObject):
        for component in components:
            if component == 'vanpy_voxceleb_gender':
                from vanpy.core.segment_classification_components.VoxcelebGenderClassifier import VoxcelebGenderClassifier
                self.components_mapper[component] = VoxcelebGenderClassifier
            if component == 'vanpy_voxceleb_age':
                from vanpy.core.segment_classification_components.VoxcelebAgeRegressor import VoxcelebAgeRegressor
                self.components_mapper[component] = VoxcelebAgeRegressor
            if component == 'vanpy_voxceleb_height':
                from vanpy.core.segment_classification_components.VoxcelebHeightRegressor import VoxcelebHeightRegressor
                self.components_mapper[component] = VoxcelebHeightRegressor
            if component == 'common_voices_gender':
                from vanpy.core.segment_classification_components.CVGenderClassifier import CVGenderClassifier
                self.components_mapper[component] = CVGenderClassifier
            elif component == 'common_voices_age':
                from vanpy.core.segment_classification_components.CVAgeClassifier import CVAgeClassifier
                self.components_mapper[component] = CVAgeClassifier
            elif component == 'speech_brain_iemocap_emotion':
                from vanpy.core.segment_classification_components.IEMOCAPEmotionClassifier import IEMOCAPEmotionClassifier
                self.components_mapper[component] = IEMOCAPEmotionClassifier
            elif component == 'vanpy_ravdess_emotion':
                from vanpy.core.segment_classification_components.RavdessEmotionClassifier import RavdessEmotionClassifier
                self.components_mapper[component] = RavdessEmotionClassifier
            elif component == 'wav2vec2adv':
                from vanpy.core.segment_classification_components.Wav2Vec2ADV import Wav2Vec2ADV
                self.components_mapper[component] = Wav2Vec2ADV
            elif component == 'wav2vec2stt':
                from vanpy.core.segment_classification_components.Wav2Vec2STT import Wav2Vec2STT
                self.components_mapper[component] = Wav2Vec2STT
            elif component == 'openai_whisper_stt':
                from vanpy.core.segment_classification_components.WhisperSTT import WhisperSTT
                self.components_mapper[component] = WhisperSTT
            elif component == 'cosine_distance_diarization':
                from vanpy.core.segment_classification_components.CosineDiarizationClassifier import CosineDiarizationClassifier
                self.components_mapper[component] = CosineDiarizationClassifier
            elif component == 'yamnet_classifier':
                from vanpy.core.segment_classification_components.YamnetClassifier import YamnetClassifier
                self.components_mapper[component] = YamnetClassifier

        super().__init__(components, config)
        self.logger.info(f'Created Classification Pipeline with {len(self.components)} components')
