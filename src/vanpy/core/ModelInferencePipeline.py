from typing import List, Dict, Optional
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline
from vanpy.core.PipelineComponent import PipelineComponent


class ClassificationPipeline(BasePipeline):
    """
    Class representing a classification pipeline, which is a specific type of BasePipeline.
    It comprises various predefined components for audio classification.

    :ivar components_mapper: Dictionary mapping component names to component classes or None.
        Each key is a string (the name of the component), and each value is either None or an instance of a class
        that inherits from PipelineComponent.
    :vartype components_mapper: Dict[str, Optional[PipelineComponent]]
    """
    components_mapper: Dict[str, Optional[PipelineComponent]] = {
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
        :type components: List[str]
        :param config: YAML configuration for the pipeline.
        :type config: YAMLObject
        """
        for component in components:
            if component == 'vanpy_voxceleb_gender':
                from vanpy.core.model_inference_components.VoxcelebGenderClassifier import VoxcelebGenderClassifier
                self.components_mapper[component] = VoxcelebGenderClassifier
            elif component == 'vanpy_voxceleb_age':
                from vanpy.core.model_inference_components.VoxcelebAgeRegressor import VoxcelebAgeRegressor
                self.components_mapper[component] = VoxcelebAgeRegressor
            elif component == 'vanpy_voxceleb_height':
                from vanpy.core.model_inference_components.VoxcelebHeightRegressor import VoxcelebHeightRegressor
                self.components_mapper[component] = VoxcelebHeightRegressor
            elif component == 'common_voices_gender':
                from vanpy.core.model_inference_components.CVGenderClassifier import CVGenderClassifier
                self.components_mapper[component] = CVGenderClassifier
            elif component == 'common_voices_age':
                from vanpy.core.model_inference_components.CVAgeClassifier import CVAgeClassifier
                self.components_mapper[component] = CVAgeClassifier
            elif component == 'speech_brain_iemocap_emotion':
                from vanpy.core.model_inference_components.IEMOCAPEmotionClassifier import IEMOCAPEmotionClassifier
                self.components_mapper[component] = IEMOCAPEmotionClassifier
            elif component == 'vanpy_ravdess_emotion':
                from vanpy.core.model_inference_components.RavdessEmotionClassifier import RavdessEmotionClassifier
                self.components_mapper[component] = RavdessEmotionClassifier
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
