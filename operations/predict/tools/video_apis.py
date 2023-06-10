import av
import numpy as np
import os
import torch

from . import _video_opt
from .video import _read_from_stream


def read_video(
    filename: str,
    start_pts=0,
    end_pts=None,
    pts_unit: str = "pts",
    return_frames=False
):
    if not os.path.exists(filename):
        raise RuntimeError(f'File not found: {filename}')

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(
            "end_pts should be larger than start_pts, got "
            "start_pts={} and end_pts={}".format(start_pts, end_pts)
        )

    info = {}
    video_frames = []

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            time_base = _video_opt.default_timebase
            if container.streams.video:
                time_base = container.streams.video[0].time_base
            elif container.streams.audio:
                time_base = container.streams.audio[0].time_base
            # video_timebase is the default time_base
            start_pts, end_pts, pts_unit = _video_opt._convert_to_sec(
                start_pts, end_pts, pts_unit, time_base)
            if container.streams.video:
                video_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate
                # guard against potentially corrupted files
                if video_fps is not None:
                    info["video_fps"] = float(video_fps)

    except av.AVError:
        pass

    vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]

    if return_frames:
        return vframes_list, video_fps

    if vframes_list:
        vframes = torch.as_tensor(np.stack(vframes_list))
    else:
        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    return vframes, video_fps
