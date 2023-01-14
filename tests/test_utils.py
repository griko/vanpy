import os
import shutil
import tempfile
import subprocess
import gdown
import unittest

from yaml import YAMLError

from vanpy.utils.utils import *


class UtilsTest(unittest.TestCase):
    def test_create_dirs_if_not_exist(self):
        dir_path = tempfile.mkdtemp()
        # Test creating new directory
        new_dir = os.path.join(dir_path, 'new_dir')
        create_dirs_if_not_exist(new_dir)
        self.assertTrue(os.path.exists(new_dir))
        # Test creating multiple new directories
        new_dir2 = os.path.join(new_dir, 'new_dir2')
        create_dirs_if_not_exist(new_dir, new_dir2)
        self.assertTrue(os.path.exists(new_dir2))
        # Test that existing directory is not overwritten
        create_dirs_if_not_exist(new_dir)
        self.assertTrue(os.path.exists(new_dir))
        shutil.rmtree(dir_path)

    def test_cut_segment(self):
        temp_dir = tempfile.mkdtemp()
        input_path = gdown.download(
            'https://drive.google.com/uc?export=download&confirm=9iBg&id=1URDocYaa0tKe3KLiFJd5ct7tsczA3mX4',
            temp_dir + '/empty.wav', quiet=True)
        output_path = cut_segment(input_path, temp_dir, (0, 0.01), 0, '_segment')
        self.assertTrue(os.path.exists(output_path))
        output_path2 = cut_segment(input_path, temp_dir, (0, 0.02), 1, '_segment')
        self.assertTrue(os.path.exists(output_path2))
        self.assertNotEqual(output_path, output_path2)
        shutil.rmtree(temp_dir)

    def test_get_audio_files_paths(self):
        temp_dir = tempfile.mkdtemp()
        input_path = gdown.download(
            'https://drive.google.com/uc?export=download&confirm=9iBg&id=1URDocYaa0tKe3KLiFJd5ct7tsczA3mX4',
            temp_dir + '/empty.wav', quiet=True)
        audio_files = get_audio_files_paths(temp_dir)
        self.assertEqual(len(audio_files), 1)
        self.assertEqual(audio_files[0], input_path)
        shutil.rmtree(temp_dir)

    def test_cached_download(self):
        # Test case where the file doesn't exist yet
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "empty.wav")
        url = "https://drive.google.com/uc?export=download&confirm=9iBg&id=1URDocYaa0tKe3KLiFJd5ct7tsczA3mX4"
        cached_download(url, file_path)
        self.assertTrue(os.path.isfile(file_path))

        # Test case where the file already exists
        file_path = os.path.join(temp_dir, "empty.wav")
        cached_download(url, file_path)
        self.assertTrue(os.path.isfile(file_path))

    def test_yaml_placeholder_replacement(self):
        test_yaml = {
            'key2': 'value2',
            'key1': '{{key2}}',
            'key3': {
                'key4': '{{key2}}'
            },
            'key5': [
                '{{key2}}'
            ]
        }
        expected_output = {
            'key2': 'value2',
            'key1': 'value2',
            'key3': {
                'key4': 'value2'
            },
            'key5': [
                'value2'
            ]
        }
        self.assertEqual(yaml_placeholder_replacement(test_yaml), expected_output)

    def test_load_config(self):
        # Test case for a valid yaml file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "config.yaml")
        with open(file_path, "w") as f:
            f.write("key: value")
        config = load_config(file_path)
        self.assertEqual(config, {"key": "value"})

        # Test case for an invalid yaml file
        file_path = os.path.join(temp_dir, "config.yaml")
        with open(file_path, "w") as f:
            f.write("key: value:")
        with self.assertRaises(YAMLError):
            load_config(file_path)

    def test_get_null_wav_path(self):
        null_wav_path = get_null_wav_path()
        self.assertTrue(os.path.isfile(null_wav_path))

