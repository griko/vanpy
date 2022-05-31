from typing import List

from yaml import YAMLObject

from core.BasePipline import BasePipeline
from core.feature_extraction_components.embedding.PyannoteEmbedding import PyannoteEmbedding


class FeatureExtractionPipeline(BasePipeline):
    components_mapper = {
        'pyannote_embedding': PyannoteEmbedding
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Feature Extraction Pipeline with {len(self.components)} components')
