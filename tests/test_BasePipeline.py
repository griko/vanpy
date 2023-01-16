import unittest
from typing import List
from unittest.mock import MagicMock, patch
from vanpy.core.BasePipeline import BasePipeline
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from yaml import YAMLObject
import pandas as pd


class TestBasePipeline(unittest.TestCase):

    class ImpBaselinePipeline(BasePipeline):
        class ImpPipelineComponent(PipelineComponent):
            def process(self, input_payload: ComponentPayload) -> ComponentPayload:
                pass

        components_mapper = {'my_new_speaker_embedding': ImpPipelineComponent}

        def __init__(self, components: List[str], config: YAMLObject):
            new_component = MagicMock(return_value=self.ImpPipelineComponent('feature_extractor', 'cool_embedding', config))
            self.components_mapper['my_new_speaker_embedding'] = new_component
            super().__init__(components, config)

    def setUp(self):
        self.base_pipeline = self.ImpBaselinePipeline(['my_new_speaker_embedding'], {'config': 'config'})  # , self.components_mapper)

    def test_init(self):
        self.assertEqual(len(self.base_pipeline.components), 1)
        self.assertIsInstance(self.base_pipeline.components[0], PipelineComponent)
        self.assertEqual(self.base_pipeline.components[0].get_name(), 'cool_embedding')
        self.assertEqual(self.base_pipeline.components[0].config, {'config': 'config'})

    def test_process(self):
        base_pipeline_comp = self.base_pipeline.components_mapper['my_new_speaker_embedding']
        base_pipeline_comp.return_value = ComponentPayload(input_path='path')
        payload = ComponentPayload(input_path='path')
        self.base_pipeline.process(payload)
        base_pipeline_comp.assert_called_once()
        self.assertEqual(payload.metadata, {'all_paths_columns': [''],
                                             'classification_columns': [],
                                             'feature_columns': [],
                                             'input_path': 'path',
                                             'meta_columns': [],
                                             'paths_column': ''})
        self.assertEqual(type(payload.df), pd.DataFrame)
        self.assertTrue(payload.df.empty)
        self.assertEqual(list(payload.df.columns), [])

    def test_get_components(self):
        class ImpPipelineComponent(PipelineComponent):
            def process(self, input_payload: ComponentPayload) -> ComponentPayload:
                pass

        components = self.base_pipeline.get_components()
        new_component = ImpPipelineComponent('feature_extractor', 'cool_embedding', {'config': 'config'})
        self.assertEqual(components[0].component_type, new_component.component_type)
        self.assertEqual(components[0].component_name, new_component.component_name)
        self.assertEqual(components[0].config, new_component.config)
        self.assertEqual(components[0].logger, new_component.logger)
