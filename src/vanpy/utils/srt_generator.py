import pandas as pd


def second_to_timecode(x: float) -> str:
    hour, x = divmod(x, 3600)
    minute, x = divmod(x, 60)
    second, x = divmod(x, 1)
    millisecond = int(x * 1000.)

    return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)


def to_srt(words_times: pd.DataFrame, start_col: str, end_col: str, text_col: str) -> str:
    def _helper(end: int) -> None:
        lines.append("%d" % section)
        lines.append(
            "%s --> %s" %
            (
                second_to_timecode(start_sec.iloc[start]),
                second_to_timecode(end_sec.iloc[end])
            )
        )
        lines.append(' '.join(x for x in words.iloc[start:(end + 1)]))
        lines.append('')

    lines = list()
    section = 0
    start = 0
    words = words_times[text_col]
    start_sec = words_times[start_col]
    end_sec = words_times[end_col]
    for k in range(1, len(words)):
        _helper(k - 1)
        start = k
        section += 1
    _helper(len(words) - 1)
    return '\n'.join(lines)


