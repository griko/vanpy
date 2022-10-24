import os
import subprocess
from typing import List, Tuple
import gdown


def create_dirs_if_not_exist(*args: str) -> None:
    for arg in args:
        os.makedirs(arg, exist_ok=True)


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
    return [f'{folder}/{f}' for f in folder_files if f.endswith(extension) and os.path.isfile(f'{folder}/{f}')]


def cached_download(url, path):
    if os.path.exists(path):
        pass
    else:
        create_dirs_if_not_exist('/'.join(path.split('/')[:-1]))
        gdown.download(url, path, quiet=True)
    return path
