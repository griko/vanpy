import unittest
import pandas as pd

from vanpy.core.ComponentPayload import ComponentPayload


class TestComponentPayload(unittest.TestCase):
    def setUp(self):
        self.input_path = 'data.csv'
        self.metadata = {'input_path': self.input_path, 'paths_column': 'path', 'all_paths_columns': ['path'],
                         'meta_columns': ['wav_converter_perf', 'vad_perf'], 'feature_columns': ['MFCC0', 'embedding_feature1'],
                         'classification_columns': ['gender']}
        self.df = pd.DataFrame({'path': ['a', 'b', 'c'], 'wav_converter_perf': [20, 30, 40], 'vad_perf': [1.2 , 5, 7],
                               'MFCC0': [37.5, 36.6, 38.4], 'embedding_feature1': [80, 75, 72], 'gender': ['M', 'F', 'F']})
        self.payload = ComponentPayload(self.input_path, self.metadata, self.df)

    def test_init(self):
        self.assertEqual(self.payload.metadata, self.metadata)
        self.assertTrue(self.payload.df.equals(self.df))

    def test_unpack(self):
        self.assertEqual(self.payload.unpack(), (self.metadata, self.df))

    def test_get_columns(self):
        self.assertEqual(self.payload.get_columns(), ['path'])
        self.assertEqual(self.payload.get_columns(all_paths_columns=True), ['path'])
        self.assertEqual(self.payload.get_columns(meta_columns=True), ['path', 'wav_converter_perf', 'vad_perf'])
        self.assertEqual(self.payload.get_columns(all_paths_columns=True, meta_columns=True), ['path', 'wav_converter_perf', 'vad_perf'])

    def test_get_declared_columns(self):
        self.assertTrue(self.payload.get_declared_columns(['feature_columns']).equals(self.df[['path', 'MFCC0', 'embedding_feature1']]))
        self.assertTrue(self.payload.get_declared_columns(['classification_columns']).equals(self.df[['path', 'gender']]))
        self.assertTrue(self.payload.get_declared_columns(['feature_columns', 'classification_columns']).equals(self.df[['path', 'MFCC0', 'embedding_feature1', 'gender']]))
