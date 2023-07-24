import json
import pandas as pd
import os

"""
Parser for Speaker_Naming_2018 dataset.
"""

def get_tt_from_time(time: str) -> float:
    """
    Returns the time in seconds from a time string.

    :param time: The time string in the format: 'hh:mm:ss,mmm'
    :return: The time in seconds.
    """
    h, m, s = time.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def get_df_from_srt(srt_path: str) -> pd.DataFrame:
    """
    Returns a pandas DataFrame from a srt file.

    :param srt_path: The path to the srt file.
    :return: A pandas DataFrame with the following columns: ['speaker', 'id', 'duration', 'text', 'seg',
    'start', 'end', 'start_tt', 'end_tt']
    """
    srt_json = json.load(open(srt_path))
    df = pd.DataFrame(srt_json, columns=['speaker', 'id', 'duration', 'text', 'seg'])
    df['start'] = df['seg'].apply(lambda x: x.split(' --> ')[0])
    df['end'] = df['seg'].apply(lambda x: x.split(' --> ')[1])
    df['start_tt'] = df['start'].apply(get_tt_from_time)
    df['end_tt'] = df['end'].apply(get_tt_from_time)
    return df

