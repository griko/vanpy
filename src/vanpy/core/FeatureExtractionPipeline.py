from typing import List, Dict, Optional
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline
from vanpy.core.PipelineComponent import PipelineComponent


class FeatureExtractionPipeline(BasePipeline):
    """
    Class representing a feature extraction pipeline, which is a specific type of BasePipeline.
    It comprises various predefined components for audio feature extraction.

    :ivar components_mapper: Dictionary mapping component names to component classes or None.
        Each key is a string (the name of the component), and each value is either None or an instance of a class
        that inherits from PipelineComponent.
    :vartype components_mapper: Dict[str, Optional[PipelineComponent]]
    """
    components_mapper: Dict[str, Optional[PipelineComponent]] = {
        'pyannote_embedding': None,
        'speechbrain_embedding': None,
        'librosa_features_extractor': None,
        'vanpy_speaker_embedding': None
    }

    def __init__(self, components: List[str], config: YAMLObject):
        """
        Initializes the FeatureExtractionPipeline object with the specified components and YAML configuration.

        The components list should be a list of strings where each string is a key in the `components_mapper`
        dictionary. The function then replaces the None value for each key in the `components_mapper` dictionary
        with the corresponding component class.

        :param components: List of names of the feature extraction components to include in this pipeline.
        :type components: List[str]
        :param config: YAML configuration for the pipeline.
        :type config: YAMLObject
        """
        for component in components:
            if component == 'pyannote_embedding':
                from vanpy.core.feature_extraction_components.PyannoteEmbedding import PyannoteEmbedding
                self.components_mapper[component] = PyannoteEmbedding
            elif component == 'speechbrain_embedding':
                from vanpy.core.feature_extraction_components.SpeechBrainEmbedding import SpeechBrainEmbedding
                self.components_mapper[component] = SpeechBrainEmbedding
            elif component == 'librosa_features_extractor':
                from vanpy.core.feature_extraction_components.LibrosaFeaturesExtractor import LibrosaFeaturesExtractor
                self.components_mapper[component] = LibrosaFeaturesExtractor
            elif component == 'vanpy_speaker_embedding':
                from vanpy.core.feature_extraction_components.VanpySpeakerEmbedding import VanpySpeakerEmbedding
                self.components_mapper[component] = VanpySpeakerEmbedding
        super().__init__(components, config)
        self.logger.info(f'Created Feature Extraction Pipeline with {len(self.components)} components')
