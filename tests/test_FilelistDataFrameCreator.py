import os
import shutil
import tempfile
import unittest

import pandas as pd
import tempfile
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.PipelineComponent import PipelineComponent
from vanpy.core.preprocess_components.FilelistDataFrameCreator import FilelistDataFrameCreator
from vanpy.utils.utils import get_audio_files_paths
from yaml import YAMLObject
from unittest import mock
import pickle

class FilelistDataFrameCreatorTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.temp_dir, 'input')
        os.makedirs(self.input_path)
        self.test_file_1 = os.path.join(self.input_path, 'test1.wav')
        self.test_file_2 = os.path.join(self.input_path, 'test2.wav')
        open(self.test_file_1, 'a').close()
        open(self.test_file_2, 'a').close()
        self.config = {
            'preprocessing': {
                'file_mapper': {
                    'load_payload': False
                }
            }
        }
        # self.payload = ComponentPayload(metadata={'input_path': self.input_path}, df=None)

    def test_process_creates_df(self):
        input_payload = ComponentPayload(metadata={'input_path': self.input_path}, df=pd.DataFrame())
        config = {'load_payload': False}
        component = FilelistDataFrameCreator(config)
        output_payload = component.process(input_payload)
        output_df = output_payload.df
        self.assertEqual(len(output_df), 2)
        self.assertEqual(output_df.columns[0], 'file_mapper_paths')
        self.assertEqual(set(output_df['file_mapper_paths'].apply(lambda x: x.split('/')[-1])), set(['test1.wav', 'test2.wav']))

    def test_process_loads_df(self):
        df = pd.DataFrame({'sample_path': [self.test_file_1, self.test_file_2], 'col1': [1, 2], 'col2': [5, 6]})
        metadata = {'paths_column': 'sample_path'}
        with tempfile.TemporaryDirectory() as tmpdirname:
            df_path = f'{tmpdirname}/df.csv'
            meta_path = f'{tmpdirname}/meta.pickle'
            df.to_csv(df_path, index=False)
            with open(meta_path, 'wb') as handle:
                pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

            df = pd.read_csv(df_path)
            with open(meta_path, 'rb') as handle:
                metadata = pickle.load(handle)
            config = {
                        'preprocessing': {
                            'file_mapper': {
                                'load_payload': True,
                                'load_df_path': df_path,
                                'load_meta_path': meta_path
                            }
                        }
                    }
            component = FilelistDataFrameCreator(config)
            output_payload = component.process(ComponentPayload(metadata=metadata, df=df))
            output_df = output_payload.df
            output_meta = output_payload.metadata
            pd.testing.assert_frame_equal(output_df, df)
            self.assertEqual(output_meta, metadata)

    def test_process_missing_input_path_metadata(self):
        input_payload = ComponentPayload(metadata={'load_payload': False, 'paths_column': 'tmp'})
        file_mapper = FilelistDataFrameCreator(self.config)
        with self.assertRaises(AttributeError) as context:
            file_mapper.process(input_payload)
            self.assertTrue(
                "The supplied ComponentPayload does not contain 'input_path', file_mapper can not be used without it" in str(
                    context.exception))