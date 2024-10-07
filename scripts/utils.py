from pathlib import Path
from itertools import pairwise


def extract_annotations(p: str | Path) -> list:
    """Reads EAF file and produces a list of (start_ms, end_ms)
    tuples where filled pauses were annotated.

    :param str | Path p: input EAF file
    :return list:list of (start_ms, end_ms) filled pauses.
    """
    from lxml import etree as ET

    doc = ET.fromstring(Path(p).read_bytes())
    ees = doc.findall(".//ALIGNABLE_ANNOTATION")
    if len(ees) == 0:
        return []
    timeline = {
        i.get("TIME_SLOT_ID"): i.get("TIME_VALUE") for i in doc.findall(".//TIME_SLOT")
    }
    return [
        (
            int(timeline[i.get("TIME_SLOT_REF1")]),
            int(timeline[i.get("TIME_SLOT_REF2")]),
        )
        for i in ees
    ]


def is_overlapping(this, other):
    if (this[0] < other[1]) and (this[1] > other[0]):
        return True
    return False


def intervals_to_frames(intervals, duration_ms: int) -> list:
    import pandas as pd

    i = pd.interval_range(start=0, end=duration_ms, freq=20)
    df = pd.DataFrame(
        data={
            "left": [ie.left for ie in i],
            "right": [ie.right for ie in i],
            "label": [0 for _ in i],
        }
    )
    for a in intervals:
        start_ms = a[0]
        end_ms = a[1]
        c = (df.right > start_ms) & (df.right < end_ms)
        df.loc[c, "label"] = 1
    return df.label.tolist()


def frames_to_intervals(frames: list) -> list[tuple]:
    """Takes the 50Hz frames [0,1,1,1,1,0,0,0...] and decodes contiguous
    sections with 1 to milliseconds [[start_ms, end_ms], ...]

    :param list frames: 50Hz frames of zeros and ones
    :return list[tuple]: list of []
    """
    import pandas as pd

    return_list = []
    ndf = pd.DataFrame(
        data={
            "millisecond": [20 * i for i in range(len(frames))],
            "frames": frames,
        }
    )

    ndf["millisecond"] = ndf.millisecond.astype(int)
    ndf = ndf.dropna()
    indices_of_change = ndf.frames.diff()[ndf.frames.diff() != 0].index.values
    for si, ei in pairwise(indices_of_change):
        if ndf.loc[si : ei - 1, "frames"].mode()[0] == 0:
            pass
        else:
            return_list.append(
                (ndf.loc[si, "millisecond"], ndf.loc[ei - 1, "millisecond"])
            )
    return return_list
