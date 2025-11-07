import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
# 指定视频文件夹路径
folder_path = "/home/user/new_Bench2Drive/output"
# folder_path = "/home/user/new_Bench2Drive/Bench2Drive/output_vad"

# 获取文件夹下所有视频文件路径
video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

# 创建 VideoFileClip 对象列表
clips = [VideoFileClip(f) for f in video_files]

# 合并视频片段
final_clip = concatenate_videoclips(clips)

# 输出合并后的视频
final_clip.write_videofile("merged_video.mp4")