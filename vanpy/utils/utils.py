import os
import subprocess
from typing import List, Tuple
import gdown
import logging

logger = logging.getLogger(f'vanpy utils')


def create_dirs_if_not_exist(*args: str) -> None:
    for arg in args:
        os.makedirs(arg, exist_ok=True)
        # logger.info(f'Created dir {arg}')


def cut_segment(input_path: str, output_dir: str, segment: Tuple[float, float], segment_id: int, separator: str) -> str:
    create_dirs_if_not_exist(output_dir)
    start, stop = segment
    f = ''.join(str(input_path).split("/")[-1].split(".")[:-1])
    output_path = f'{output_dir}/{f}{separator}{segment_id}.wav'
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start}", "-to", f"{stop}", "-y", "-i",
                    f"{input_path}", "-ab", "256k", "-ac", "1", "-ar", "16k", output_path, '-dn',
                    '-ignore_unknown', '-sn'])
    return output_path


def get_audio_files_paths(folder: str, extension: str = '') -> List[str]:
    folder_files = os.listdir(folder)
    logger.info(f'Finished listdir on {folder}')
    return [f'{folder}/{f}' for f in folder_files if f.endswith(extension) and os.path.isfile(f'{folder}/{f}')]


def cached_download(url, path):
    if os.path.exists(path):
        pass
    else:
        create_dirs_if_not_exist('/'.join(path.split('/')[:-1]))
        gdown.download(url, path, quiet=True)
    return path


def yaml_placeholder_replacement(full, val=None, initial=True):
    val = val or full if initial else val
    if isinstance(val, dict):
        for k, v in val.items():
            val[k] = yaml_placeholder_replacement(full, v, False)
    elif isinstance(val, list):
        for idx, i in enumerate(val):
            val[idx] = yaml_placeholder_replacement(full, i, False)
    elif isinstance(val, str):
        while "{{" in val and "}}" in val:
            val = full[val.split("}}")[0].split("{{")[1]] + ''.join(val.split("}}")[1:])

    return val


def get_null_wav_path():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "empty.wav")
    if not os.path.isfile(path):
        gdown.download('https://drive.google.com/uc?export=download&confirm=9iBg&id=1URDocYaa0tKe3KLiFJd5ct7tsczA3mX4', path, quiet=True)
    return path
