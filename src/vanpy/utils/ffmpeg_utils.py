import subprocess
import pandas as pd
from vanpy.utils.utils import create_dirs_if_not_exist
from tqdm import tqdm


def cut_segments(input_file_path: str, df: pd.DataFrame, output_dir: str, offset: float = 0.0, play_speed_multiplier: float = 1.0, start_tt_column_name: str = 'start_tt', end_tt_column_name: str = 'end_tt', path_column_name: str = 'input_path') -> pd.DataFrame:
    """
    Cut segments from an audio file using FFMPEG.

    :param input_file_path: The path of the input audio file.
    :param df: A DataFrame containing the start and end timestamps of the segments to be cut.
    :param output_dir: The path of the output directory.
    :param offset: The offset to be applied to the start and end timestamps.
    :param play_speed_multiplier: The play speed multiplier to be applied to the start and end timestamps.
    :param start_tt_column_name: The name of the column containing the start timestamps.
    :param end_tt_column_name: The name of the column containing the end timestamps.
    :param path_column_name: The name of the column containing the input audio file path.
    :return: An extended DataFrame containing the paths of the cut segments.
    """
    create_dirs_if_not_exist(output_dir)
    ffmpeg_config = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                     "input_file", "-ss", "start_tt", "-to", "end_tt", "-vn", "output_file", '-dn', '-ignore_unknown', '-sn']
    input_file_idx = ffmpeg_config.index("input_file")
    output_file_idx = ffmpeg_config.index("output_file")
    start_tt_idx = ffmpeg_config.index("start_tt")
    end_tt_idx = ffmpeg_config.index("end_tt")

    ffmpeg_config[input_file_idx] = f"{input_file_path}"

    for index, row in tqdm(df.iterrows()):
        output_filename: str = input_file_path.split('/')[-1].split('\\')[-1].split('.')[0] + f'_{index}.wav'
        start_tt = row[start_tt_column_name]
        end_tt = row[end_tt_column_name]
        if type(start_tt) == str:
            start_tt = start_tt.replace(',', '.')
        else:
            start_tt *= play_speed_multiplier
            start_tt -= offset
        if type(end_tt) == str:
            end_tt = end_tt.replace(',', '.')
        else:
            end_tt *= play_speed_multiplier
            end_tt -= offset
        ffmpeg_config[output_file_idx] = f'{output_dir}/{output_filename}'
        ffmpeg_config[start_tt_idx] = f"{start_tt}"
        ffmpeg_config[end_tt_idx] = f"{end_tt}"
        subprocess.run(ffmpeg_config)
        df.loc[index, path_column_name] = f'{output_dir}/{output_filename}'
    return df
