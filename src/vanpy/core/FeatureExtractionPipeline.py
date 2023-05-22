from typing import List
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline

class FeatureExtractionPipeline(BasePipeline):
    components_mapper = {
        'pyannote_embedding': None,
        'speechbrain_embedding': None,
        'librosa_features_extractor': None,
        'vanpy_speaker_embedding': None
    }

    def __init__(self, components: List[str], config: YAMLObject):
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
