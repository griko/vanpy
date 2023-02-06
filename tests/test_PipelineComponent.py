import tempfile
from unittest import TestCase
import unittest
import unittest.mock
import io
import pandas as pd
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.core.ComponentPayload import ComponentPayload

class TestPipelineComponent(TestCase):
    class ImpPipelineComponent(PipelineComponent):
        def process(self, input_payload: ComponentPayload) -> ComponentPayload:
            pass

    def setUp(self):
        self.component_type = "example_component"
        self.component_name = "example"
        temp_dir = tempfile.mkdtemp()
        self.config = {'example_component': {
                           'example': {"log_each_x_records": 1, "records_count": 10, "save_payload": True, "intermediate_payload_path": temp_dir}
                        }
                    }
        self.pipeline_component = self.ImpPipelineComponent(self.component_type, self.component_name, self.config)
        self.input_payload = ComponentPayload(input_path=temp_dir, df=pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}))

    def test_process_abstract_method(self):
        with self.assertRaises(TypeError) as context:
            pipeline_component = PipelineComponent("test_type", "test_name", {})
        self.assertTrue("Can't instantiate abstract class" in str(context.exception))

    def test_import_config(self):
        yaml_config = {'test_type': {'test_name': {'param1': 'value1', 'param2': 'value2'}}}
        pipeline_component = self.ImpPipelineComponent("test_type", "test_name", yaml_config)
        subset = {k: v for k, v in pipeline_component.config.items() if k in {'param1': 'value1', 'param2': 'value2'}}
        self.assertDictEqual(subset, {'param1': 'value1', 'param2': 'value2'})

    def test_get_logger(self):
        pipeline_component = self.ImpPipelineComponent("test_type", "test_name", {})
        self.assertEqual(pipeline_component.logger.name, 'test_type - test_name')

    def test_get_name(self):
        pipeline_component = self.ImpPipelineComponent("test_type", "test_name", {})
        self.assertEqual(pipeline_component.get_name(), 'test_name')

    # def test_save_intermediate_payload(self):
    #     pipeline_component = self.ImpPipelineComponent("test_type", "test_name", {"save_payload_periodicity": 2})
    #     input_payload = ComponentPayload(df=pd.DataFrame({"col1": [1, 2, 3, 4, 5]}))
    #     pipeline_component.save_intermediate_payload(1, input_payload)
    #     pipeline_component.save_intermediate_payload(2, input_payload)
    #     pipeline_component.save_intermediate_payload(3, input_payload)
    #     with patch.object(PipelineComponent, 'save_component_payload') as mock_method:
    #         pipeline_component.save_intermediate_payload(3, input_payload)
    #         self.assertEqual(mock_method.call_count, 2)

    # @unittest.mock.patch(new_callable=io.StringIO)
    # def test_latent_info_log(self, mock_stdout):
    #     self.pipeline_component.latent_info_log("example message", 2)
    #     self.assertEqual(mock_stdout.getvalue(), "example message")
    #     self.assertEqual(self.pipeline_component.logger.info.call_count, 1)
    #     self.pipeline_component.latent_info_log("example message", 3, last_item=True)
    #     self.assertEqual(self.pipeline_component.logger.info.call_count, 2)
    #     self.pipeline_component.latent_info_log("example message", 4)
    #     self.assertEqual(self.pipeline_component.logger.info.call_count, 2)