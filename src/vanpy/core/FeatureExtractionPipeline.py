from typing import List
from yaml import YAMLObject
from vanpy.core.BasePipeline import BasePipeline
from vanpy.core.feature_extraction_components.LibrosaFeaturesExtractor import LibrosaFeaturesExtractor
from vanpy.core.feature_extraction_components.PyannoteEmbedding import PyannoteEmbedding
from vanpy.core.feature_extraction_components.SpeechBrainEmbedding import SpeechBrainEmbedding


class FeatureExtractionPipeline(BasePipeline):
    components_mapper = {
        'pyannote_embedding': PyannoteEmbedding,
        'speechbrain_embedding': SpeechBrainEmbedding,
        'librosa_features_extractor': LibrosaFeaturesExtractor
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Feature Extraction Pipeline with {len(self.components)} components')
