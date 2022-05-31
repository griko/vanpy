from typing import List
from yaml import YAMLObject
from core.BasePipline import BasePipeline
from core.segment_classification_components.age.CVAgeClassifier import CVAgeClassifier
from core.segment_classification_components.gender.CVGenderClassifier import CVGenderClassifier


class ClassificationPipeline(BasePipeline):
    components_mapper = {
        'common_voices_gender': CVGenderClassifier,
        'common_voices_age': CVAgeClassifier
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Classification Pipeline with {len(self.components)} components')
