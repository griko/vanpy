import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import gdown
import pandas as pd
from vanpy.core.PipelineComponent import ComponentPayload
from vanpy.core.preprocess_components.PyannoteVAD import PyannoteVAD
from config import config_test_PyannoteVAD

class TestPyannoteVAD(unittest.TestCase):

    def setUp(self):
        self.config = config_test_PyannoteVAD
        self.vad = PyannoteVAD(self.config)

    @patch.object(PyannoteVAD, 'get_voice_segments')
    def test_process_item(self, MockGetVoiceSegments):
        # Download specific audio file
        temp_dir = tempfile.mkdtemp()
        input_path = gdown.download(
            'https://drive.google.com/uc?export=download&confirm=9iBg&id=1y3Cj0PLMz3DloEYW0fP4Iy-6xLy1rOj-',
            temp_dir + '/test_audio.wav', quiet=True
        )
        MockGetVoiceSegments.return_value = [(0.5, 2.0), (3.0, 5.0)]
        f = input_path
        p_df = pd.DataFrame()
        processed_path = 'segmented_audio'
        input_column = 'input_audio'
        output_dir = '/output'

        new_p_df = self.vad.process_item(f, p_df, processed_path, input_column, output_dir)

        self.assertEqual(len(new_p_df), 2)  # two segments should be added
        self.assertEqual(new_p_df.loc[0, processed_path], f'{output_dir}/test_audio_0.wav')  # Update with expected value
        self.assertEqual(new_p_df.loc[0, input_column], f'{input_path}')

        # Cleanup
        shutil.rmtree(temp_dir)

    @patch.object(PyannoteVAD, 'process_item')
    @patch.object(PyannoteVAD, 'load_model')
    def test_process(self, MockLoadModel, MockProcessItem):
        MockLoadModel.return_value = None  # assume model is loaded successfully
        MockProcessItem.return_value = pd.DataFrame({'segmented_audio': ['test_audio_0.wav'], 'input_audio': ['test_audio.wav']})

        metadata = {'paths_column': 'input_audio'}
        df = pd.DataFrame({'input_audio': ['test_audio.wav']})
        payload = ComponentPayload(metadata=metadata, df=df)
        new_payload = self.vad.process(payload)

        new_df = new_payload.df
        self.assertEqual(new_df.loc[0, 'segmented_audio'], 'test_audio_0.wav')  # Update with expected value

if __name__ == '__main__':
    unittest.main()