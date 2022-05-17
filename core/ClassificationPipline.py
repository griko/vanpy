from typing import List, Tuple

from yaml import YAMLObject

from core.BasePipline import BasePipeline
from core.speaker_classification_components.gender.CVGenderClassifier import CVGenderClassifier
import pandas as pd


class ClassificationPipeline(BasePipeline):
    components_mapper = {
        'common_voices_gender': CVGenderClassifier
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Classification Pipeline with {len(self.components)} components')