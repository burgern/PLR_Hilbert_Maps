from config import PATH_LOG

import os
import moviepy.video.io.ImageSequenceClip
folder_name = 'lhmc_test_v030_n_100000_size_5_lr_0.02_batchsize_32_epochs_1'
image_folder = os.path.join(PATH_LOG, folder_name)
fps = 3
image_files_1 = [os.path.join(image_folder, img)
               for img in os.listdir(image_folder)
               if 'iteration' in img]
image_files_1.sort()
image_files_data.extend(image_files_1)

image_files_2 = [os.path.join(image_folder, img)
               for img in os.listdir(image_folder)
               if 'viewpoint' in img]
image_files_2.sort()

clip1 = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files_1, fps=fps)
clip1.write_videofile('clip1.mp4')
# clip2 = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files_1, fps=fps)
# clip2.write_videofile('clip2.mp4')
#
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# clip1 = VideoFileClip("clip1.mp4")
# clip1.resize( (640, 480) )
# clip2 = VideoFileClip("clip2.mp4")
# clip2.resize( (640, 480) )
# final_clip = concatenate_videoclips([clip1, clip2])
# final_clip.write_videofile("clip_concatenated.mp4")
#
#
# from moviepy.editor import VideoFileClip, clips_array, vfx
# clip1 = VideoFileClip("clip1.mp4").margin(10) # add 10px contour
# clip2 = VideoFileClip("clip2.mp4").margin(10)
# # clip3 = clip1.fx( vfx.mirror_y)
# # clip4 = clip1.resize(0.60) # downsize 60%
# final_clip = clips_array([[clip1, clip2]])
# # final_clip.resize(width=480).write_videofile("my_stack.mp4")
# final_clip.write_videofile("clip_concatenated.mp4")
