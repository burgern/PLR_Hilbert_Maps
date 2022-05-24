from functools import wraps
import time
import os
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoFileClip
from config import PATH_PLR
from typing import Optional, Tuple
from pathlib import Path


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def create_video_stream(image_folder: str, fps: int = 1) -> str:
    image_files = [os.path.join(image_folder, img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".png")]
    image_files.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files,
                                                                fps=fps)
    video_path = os.path.join(Path(image_folder).parent, f"video.mp4")
    clip.write_videofile(video_path)
    return video_path


def create_gif_from_mp4(video_path: str, clip_time: Optional[Tuple] = None) -> \
        str:
    clip = VideoFileClip(video_path)
    if clip_time is not None:
        clip = clip.subclip(clip_time[0], clip_time[1])
    save_path = os.path.join(Path(video_path).parent, "video.gif")
    clip.write_gif(save_path)
    return save_path
