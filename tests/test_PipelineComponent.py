import tempfile
from unittest import TestCase
import unittest
import unittest.mock
import io
import pandas as pd
from src.vanpy.core.PipelineComponent import PipelineComponent
from src.vanpy.core.ComponentPayload import ComponentPayload

class TestPipelineComponent(TestCase):
    def setUp(self):
        self.component_type = "example_component"
        self.component_name = "example"
        temp_dir = tempfile.mkdtemp()
        self.config = {'example_component': {
                           'example': {"log_each_x_records": 1, "items_in_paths_list": 10, "save_payload": True, "intermediate_payload_path": temp_dir}
                        }
                    }

        class ImpPipelineComponent(PipelineComponent):
            def process(self, input_payload: ComponentPayload) -> ComponentPayload:
                pass
        self.pipeline_component = ImpPipelineComponent(self.component_type, self.component_name, self.config)
        self.input_payload = ComponentPayload(input_path=temp_dir, df=pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}))

    # @unittest.mock.patch(new_callable=io.StringIO)
    # def test_latent_info_log(self, mock_stdout):
    #     self.pipeline_component.latent_info_log("example message", 2)
    #     self.assertEqual(mock_stdout.getvalue(), "example message")
    #     self.assertEqual(self.pipeline_component.logger.info.call_count, 1)
    #     self.pipeline_component.latent_info_log("example message", 3, last_item=True)
    #     self.assertEqual(self.pipeline_component.logger.info.call_count, 2)
    #     self.pipeline_component.latent_info_log("example message", 4)
    #     self.assertEqual(self.pipeline_component.logger.info.call_count, 2)