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


def create_video_stream(image_folder: str, fps: int = 1):
    image_folder_abs = os.path.join(PATH_PLR, image_folder)
    image_files = [os.path.join(image_folder_abs, img)
                   for img in os.listdir(image_folder_abs)
                   if img.endswith(".png")]
    image_files.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files,
                                                                fps=fps)
    clip.write_videofile(os.path.join(image_folder_abs, f"video.mp4"))


def create_gif_from_mp4(video_path: str, clip_time: Optional[Tuple] = None):
    video_path = os.path.join(PATH_PLR, video_path)
    clip = VideoFileClip(video_path)
    if clip_time is not None:
        clip = clip.subclip(clip_time[0], clip_time[1])
    save_path = os.path.join(Path(video_path).parent.absolute(), "video.gif")
    clip.write_gif(save_path)
