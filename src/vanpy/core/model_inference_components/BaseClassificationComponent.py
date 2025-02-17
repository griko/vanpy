from abc import ABC
from typing import Dict, List, Tuple

from yaml import YAMLObject
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.utils.utils import get_audio_files_paths
import pandas as pd


class BaseClassificationComponent(PipelineComponent, ABC):
    """
    Abstract base class for all classification components.

    This class provides common functionality for classification components,
    including feature list management and column naming conventions.

    :ivar classification_column_name: Name of the classification output column.
    """
    def __init__(self, component_type: str, component_name: str, yaml_config: YAMLObject):
        """
        Initialize the BaseClassificationComponent.

        :param component_type: Type of the component.
        :param component_name: Name of the component.
        :param yaml_config: Configuration parameters for the component.
        """
        super().__init__(component_type=component_type, component_name=component_name,
                         yaml_config=yaml_config)
        self.classification_column_name = self.config.get('classification_column_name',
                                                          f'{self.component_name}_classification')

    def build_requested_feature_list(self):
        """
        Build a list of required features based on configuration.

        Processes the feature list configuration to create a complete list of 
        required features. Handles both single features and range-based feature sets.

        :return: List of feature names required by the component.
        :raises AttributeError: If a multiple-index feature is improperly configured.
        """
        features_list = []
        if 'features_list' in self.config:
            for feature in self.config['features_list']:
                if isinstance(feature, str):
                    features_list.append(feature)
                elif isinstance(feature, dict):
                    key = tuple(feature.keys())[0]
                    if 'start_index' not in feature[key] or 'stop_index' not in feature[key]:
                        raise AttributeError('Invalid form of multiple-index feature. You have to supply start_index and stop_index')
                    for i in range(int(feature[key]['start_index']), int(feature[key]['stop_index'])):
                        features_list.append(f'{i}_{key}')
        return features_list