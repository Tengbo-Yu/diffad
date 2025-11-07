import cv2
import os
import numpy as np
import json
from tqdm import trange
from moviepy.editor import VideoFileClip, concatenate_videoclips
#include <filesystem>  


def create_video(input_folder, output_folder, fps, font_scale, text_color, text_position):
    # print(os.path.join(images_folder, 'rgb_front'))
    
    for clip in os.listdir(input_folder):
        images_folder = os.path.join(input_folder, clip)
        print(images_folder)
        # 1600 * 900
        images_front_left = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front_left')) if img.endswith(".jpg") or img.endswith(".png")]
        images_front_left.sort()
        
        images_front = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front')) if img.endswith(".jpg") or img.endswith(".png")]
        images_front.sort()
        
        images_front_right = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front_right')) if img.endswith(".jpg") or img.endswith(".png")]
        images_front_right.sort()
        
        images_back_left = [img for img in os.listdir(os.path.join(images_folder, 'rgb_back_left')) if img.endswith(".jpg") or img.endswith(".png")]
        images_back_left.sort()
        
        images_back = [img for img in os.listdir(os.path.join(images_folder, 'rgb_back')) if img.endswith(".jpg") or img.endswith(".png")]
        images_back.sort()
        
        images_back_right = [img for img in os.listdir(os.path.join(images_folder, 'rgb_back_right')) if img.endswith(".jpg") or img.endswith(".png")]
        images_back_right.sort()
        
        # 512 * 512
        images_bev = [img for img in os.listdir(os.path.join(images_folder, 'bev')) if img.endswith(".jpg") or img.endswith(".png")]
        images_bev.sort()
        
        # for diffad
        images_pred_bev = [img for img in os.listdir(os.path.join(images_folder, 'pred_bev')) if img.endswith(".jpg") or img.endswith(".png")]
        images_pred_bev.sort()

        frame = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), images_front_left[0]))
        height, width, layers = frame.shape
        height = int(height/4)
        width = int(width/4)
        
        
        output_video = os.path.join(output_folder, clip+".mp4")
        if os.path.exists(output_video) or len(images_front_left)==0: 
            continue
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video = cv2.VideoWriter(output_video, fourcc, fps, (3*width+2*height, 2*height))
        video = cv2.VideoWriter(output_video, fourcc, fps, (3*width+2*height+height, 2*height))

        for i in trange(1, len(images_front_left)):
            image_front_left = images_front_left[i]
            image_front = images_front[i]
            image_front_right = images_front_right[i]
            image_back_left = images_back_left[i]
            image_back = images_back[i]
            image_back_right = images_back_right[i]
            image_bev = images_bev[i]
            image_pred_bev = images_pred_bev[i]
            
            image_front_left = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front_left'), image_front_left))
            cv2.putText(image_front_left, "front_left", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            
            image_front = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), image_front))
            cv2.putText(image_front, "front", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            
            image_front_right = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front_right'), image_front_right))
            cv2.putText(image_front_right, "front_right", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            
            image_back_left = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_back_left'), image_back_left))
            cv2.putText(image_back_left, "back_left", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            
            image_back = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_back'), image_back))
            cv2.putText(image_back, "back", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            
            image_back_right = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_back_right'), image_back_right))
            cv2.putText(image_back_right, "back_right", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            
            image_bev = cv2.imread(os.path.join(os.path.join(images_folder, 'bev'), image_bev))
            cv2.putText(image_bev, "bev", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            
            ## for diffad
            image_pred_bev = cv2.imread(os.path.join(os.path.join(images_folder, 'pred_bev'), image_pred_bev))
            cv2.putText(image_pred_bev, "pred_bev", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            
            image_front_left = cv2.resize(image_front_left, (width, height))
            image_front = cv2.resize(image_front, (width, height))
            image_front_right = cv2.resize(image_front_right, (width, height))
            image_back_left = cv2.resize(image_back_left, (width, height))
            image_back = cv2.resize(image_back, (width, height))
            image_back_right = cv2.resize(image_back_right, (width, height))
            image_bev = cv2.resize(image_bev, (2*height, 2*height))
            ## for diffad
            image_pred_bev = cv2.resize(image_pred_bev, (height, 2*height))
            
            # 水平拼接
            top_row = np.hstack((image_front_left, image_front, image_front_right))
            bottom_row = np.hstack((image_back_left, image_back, image_back_right))
            
            # 垂直拼接
            combined_image = np.vstack((top_row, bottom_row))
            
            
            # img = np.hstack((combined_image, image_bev))
            img = np.hstack((combined_image, image_bev, image_pred_bev))
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            f = open(os.path.join(images_folder, f'meta/{i:04}.json'), 'r')
            meta = json.load(f)
            steer = float(meta['steer'])
            throttle = float(meta['throttle'])
            brake = float(meta['brake'])
            # command = float(meta['command'])
            #    = ["VOID", "LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT",  "CHANGE LANE RIGHT",]
            speed = float(meta['speed'])
            text = f'speed: {round(speed,2)}, steer: {round(steer,2)}, throttle: {round(throttle,2)}, brake: {round(brake,2)}'#, command: {command_list[int(command)]}'
            
            # cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            video.write(img)
        video.release()


def merge_video(folder_path):
    # 指定视频文件夹路径
    # folder_path = "/home/user/new_Bench2Drive/output"
    # # folder_path = "/home/user/new_Bench2Drive/Bench2Drive/output_vad"

    # 获取文件夹下所有视频文件路径
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    # 创建 VideoFileClip 对象列表
    clips = [VideoFileClip(f) for f in video_files]

    # 合并视频片段
    final_clip = concatenate_videoclips(clips)

    # 输出合并后的视频
    final_clip.write_videofile(folder_path + "/merged_video.mp4")
    

# images_folder = '/home/user/new_Bench2Drive/Bench2Drive/eval_bench2drive220_vad_traj/bench2drive220_0_vad_traj_RouteScenario_1773_rep0_Town12_ParkedObstacle_1_25_08_20_14_57_45'
# input_folder = '/home/user/new_Bench2Drive/Bench2Drive/eval_bench2drive220_1_vad_traj_2'
input_folder = '/home/user/new_Bench2Drive/Bench2Drive/eval_bench2drive220_1_diffad-fp-0.5-plan-3s-dpm-10_traj'
output_folder = './output'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
fps = 10
font_scale = 1
text_color = (255, 255, 255)
text_position = (50, 50)

create_video(input_folder, output_folder, fps, font_scale, text_color, text_position)
merge_video(output_folder)