from typing import List

from yaml import YAMLObject

from core.BasePipline import BasePipeline
from core.speaker_classification_components.gender.CVGenderClassifier import CVGenderClassifier
import pandas as pd


class SpeakerClassificationPipeline(BasePipeline):
    components_mapper = {
        'common_voices_gender': CVGenderClassifier
    }

    def __init__(self, components: List[str], config: YAMLObject):
        super().__init__(components, config)
        self.logger.info(f'Created Speaker Classification Pipeline with {len(self.components)} components')

    def process(self) -> pd.DataFrame:
        last_output_dir = ''
        for component in self.components:
            self.logger.info(f'Processing with {component.get_name()}')
            last_output_dir = component.process()
        return last_output_dir