from typing import List, Dict, Optional
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline
from vanpy.core.PipelineComponent import PipelineComponent


class ClassificationPipeline(BasePipeline):
    """
    Pipeline for audio classification and analysis tasks.

    Manages components that perform various audio analysis tasks including:
    - Speaker characteristics (gender, age, height)
    - Emotion detection
    - Speech-to-text
    - Speaker diarization
    - Sound event classification

    :ivar components_mapper: Maps classifier names to their implementing classes.
                           Supports a wide range of classification tasks and models.
    """
    components_mapper: Dict[str, Optional[PipelineComponent]] = {
        'vanpy_gender': None,
        'vanpy_age': None,
        'vanpy_height': None,
        'vanpy_emotion': None,
        # 'common_voices_gender': None,
        # 'common_voices_age': None,
        'speech_brain_iemocap_emotion': None,
        'wav2vec2adv': None,
        'wav2vec2stt': None,
        'openai_whisper_stt': None,
        'cosine_distance_diarization': None,
        'agglomerative_clustering_diarization': None,
        'gmm_clustering_diarization': None,
        'yamnet_classifier': None
    }

    def __init__(self, components: List[str], config: YAMLObject):
        """
        Initializes the ClassificationPipeline object with the specified components and YAML configuration.

        The components list should be a list of strings where each string is a key in the `components_mapper`
        dictionary. The function then replaces the None value for each key in the `components_mapper` dictionary
        with the corresponding component class.

        :param components: List of names of the classification components to include in this pipeline.
        :param config: YAML configuration for the pipeline.
        """
        for component in components:
            if component == 'vanpy_gender':
                from vanpy.core.model_inference_components.VanpyGenderClassifier import VanpyGenderClassifier
                self.components_mapper[component] = VanpyGenderClassifier
            elif component == 'vanpy_age':
                from vanpy.core.model_inference_components.VanpyAgeRegressor import VanpyAgeRegressor
                self.components_mapper[component] = VanpyAgeRegressor
            elif component == 'vanpy_height':
                from vanpy.core.model_inference_components.VanpyHeightRegressor import VanpybHeightRegressor
                self.components_mapper[component] = VanpybHeightRegressor
            elif component == 'vanpy_emotion':
                from vanpy.core.model_inference_components.VanpyEmotionClassifier import VanpyEmotionClassifier
                self.components_mapper[component] = VanpyEmotionClassifier
            elif component == 'speech_brain_iemocap_emotion':
                from vanpy.core.model_inference_components.IEMOCAPEmotionClassifier import IEMOCAPEmotionClassifier
                self.components_mapper[component] = IEMOCAPEmotionClassifier
            elif component == 'wav2vec2adv':
                from vanpy.core.model_inference_components.Wav2Vec2ADV import Wav2Vec2ADV
                self.components_mapper[component] = Wav2Vec2ADV
            elif component == 'wav2vec2stt':
                from vanpy.core.model_inference_components.Wav2Vec2STT import Wav2Vec2STT
                self.components_mapper[component] = Wav2Vec2STT
            elif component == 'openai_whisper_stt':
                from vanpy.core.model_inference_components.WhisperSTT import WhisperSTT
                self.components_mapper[component] = WhisperSTT
            elif component == 'cosine_distance_diarization':
                from vanpy.core.model_inference_components.CosineDistanceClusterer import CosineDistanceClusterer
                self.components_mapper[component] = CosineDistanceClusterer
            elif component == 'agglomerative_clustering_diarization':
                from vanpy.core.model_inference_components.AgglomerativeClusterer import AgglomerativeClusterer
                self.components_mapper[component] = AgglomerativeClusterer
            elif component == 'gmm_clustering_diarization':
                from vanpy.core.model_inference_components.GMMClusterer import GMMClusterer
                self.components_mapper[component] = GMMClusterer
            elif component == 'yamnet_classifier':
                from vanpy.core.model_inference_components.YamnetClassifier import YamnetClassifier
                self.components_mapper[component] = YamnetClassifier

        super().__init__(components, config)
        self.logger.info(f'Created Classification Pipeline with {len(self.components)} components')
