import os
import subprocess
from typing import List, Tuple
import gdown


def create_dirs_if_not_exist(*args: str) -> None:
    for arg in args:
        os.makedirs(arg, exist_ok=True)


def cut_segment(input_path: str, output_dir: str, segment: Tuple[float, float], segment_id: int) -> str:
    create_dirs_if_not_exist(output_dir)
    start, stop = segment
    f = ''.join(str(input_path).split("/")[-1].split(".")[:-1])
    output_path = f'{output_dir}/{f}_{segment_id}.wav'
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start}", "-to", f"{stop}", "-y", "-i",
                    f"{input_path}", "-ab", "256k", "-ac", "1", "-ar", "16k", output_path, '-dn',
                    '-ignore_unknown', '-sn'])
    return output_path


# def cut_by_segments(input_path:str, output_path:str, segments_list:List[Tuple[float,float]]) -> None:
#     create_dirs_if_not_exist(output_path)
#     if segments_list:
#       for i, segment in enumerate(segments_list):
#           start, stop = segment
#           f = ''.join(str(input_path).split("/")[-1].split(".")[:-1])
#           subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start}", "-to", f"{stop}", "-y",
#           "-i", f"{input_path}", "-ab", "256k", "-ac", "1", "-ar", "16k", f'{output_path}/{f}_{i}.wav', '-dn',
#           '-ignore_unknown', '-sn'])

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
