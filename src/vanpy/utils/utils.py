import os
import subprocess
from typing import List, Tuple, Dict
import gdown
import logging
import yaml

logger = logging.getLogger(f'vanpy utils')


def create_dirs_if_not_exist(*args: str) -> None:
    """
    Create directories if they do not exist.
    :param args: directory paths
    """
    for arg in args:
        os.makedirs(arg, exist_ok=True)
        # logger.info(f'Created dir {arg}')


def cut_segment(input_path: str, output_dir: str, segment: Tuple[float, float], segment_id: int, separator: str,
                keep_only_first_segment: bool) -> str:
    """
    Cut a segment of audio from a given file.
    :param input_path: path to audio file
    :param output_dir: directory where the segmented audio file should be stored
    :param segment: start and end time of the segment in seconds
    :param segment_id: id of the segment
    :param separator: separator to use in the output file name
    :param keep_only_first_segment: indicates if there is a single segment cut
    :return: path of the segmented audio file
    """
    create_dirs_if_not_exist(output_dir)
    start, stop = segment
    f = ''.join(str(input_path).split("/")[-1].split(".")[:-1])
    segment_suffix = f'{separator}{segment_id}' if not keep_only_first_segment else ""
    output_path = f'{output_dir}/{f}{segment_suffix}.wav'
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start}", "-to", f"{stop}", "-y", "-i",
                    f"{input_path}", "-ab", "256k", "-ac", "1", "-ar", "16k", '-dn',
                    '-ignore_unknown', '-sn',  output_path])
    return output_path


def get_audio_files_paths(folder: str, extension: str = '') -> List[str]:
    """
    Get a list of audio files in a given folder.
    :param folder: path of the folder
    :param extension: extension of the audio files
    :return: list of audio file paths
    """
    folder_files = os.listdir(folder)
    logger.info(f'Finished listdir on {folder}')
    return [f'{folder}/{f}' for f in folder_files if f.endswith(extension) and os.path.isfile(f'{folder}/{f}')]


def cached_download(url, path) -> str:
    """
    Download a file from a given URL and save it to a specified path. If the file already exists, skip the download.
    :param url: URL of the file
    :param path: path to save the file
    :return: path of the downloaded file
    """
    separator = "/"  # os.sep
    if os.path.exists(path):
        pass
    else:
        folder_name = separator.join(path.split(separator)[:-1])
        if folder_name != '':
            create_dirs_if_not_exist(folder_name)
        gdown.download(url, path, quiet=True)
    return path


def yaml_placeholder_replacement(full, val=None, initial=True) -> yaml.YAMLObject:
    """
    Replace placeholders in a YAML file with their corresponding values.
    :param full: full YAML file
    :param val: current value being processed
    :param initial: flag to indicate if this is the initial call of the function
    :return: YAML file with placeholders replaced
    """
    val = val or full if initial else val
    if isinstance(val, dict):
        for k, v in val.items():
            val[k] = yaml_placeholder_replacement(full, v, False)
    elif isinstance(val, list):
        for idx, i in enumerate(val):
            val[idx] = yaml_placeholder_replacement(full, i, False)
    elif isinstance(val, str):
        while "{{" in val and "}}" in val:
            val = str(full[val.split("}}")[0].split("{{")[1]]) + ''.join(val.split("}}")[1:])

    return val


def load_config(config_yaml_path: str = 'pipeline.yaml') -> Dict:
    """
    Load a YAML configuration file and replace any placeholders with their corresponding values.
    :param config_yaml_path: path of the configuration file
    :return: configuration as a dictionary
    """
    with open(config_yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml_placeholder_replacement(config)
    return config


def get_null_wav_path() -> str:
    """
    Get the path of a null wav file.
    :return: path of the null wav file
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "empty.wav")
    if not os.path.isfile(path):
        gdown.download('https://drive.google.com/uc?export=download&confirm=9iBg&id=1URDocYaa0tKe3KLiFJd5ct7tsczA3mX4',
                       path, quiet=True)
    return path

def concat_audio_files_in_dir(input_dir, output_path, extension='.wav', sr=16000, required_substring=''):
    """
    Concatenate audio files that contain "required_substring" in a directory into a one audio file.
    :param input_dir: The directory containing the audio files to be concatenated.
    :param output_path: The path where the concatenated audio file will be saved.
    :param extension: The file extension of the audio files in the directory to be concatenated. Defaults to '.wav'.
    :param sr: The sampling rate of the concatenated audio file. Defaults to 16000.
    :param required_substring: The required substring in the audio file names to be concatenated. Defaults to ''.
    """
    import os
    import librosa
    import numpy as np
    import soundfile as sf

    files = os.listdir(input_dir)
    files = [f for f in files if f.endswith(extension) and required_substring in f]
    files = [os.path.join(input_dir, f) for f in files]
    audio = []
    for f in files:
        y, sr = librosa.load(f, sr=sr)
        audio.append(y)
    audio = np.concatenate(audio)
    sf.write(output_path, audio, samplerate=sr)
