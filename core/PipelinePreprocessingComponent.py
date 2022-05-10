from abc import ABC
from dataclasses import dataclass
from core.PiplineComponent import PipelineComponent
from yaml import YAMLObject


@dataclass
class PipelinePreprocessingComponent(PipelineComponent, ABC):
    input_dir: str

    def __init__(self, component_type: str, component_name: str, yaml_config: YAMLObject, input_dir: str):
        super().__init__(component_type, component_name, yaml_config)
        self.input_dir = input_dir

    def get_input_dir(self) -> str:
        return self.input_dir
