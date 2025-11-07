import os
import glob
from tqdm import tqdm
import cv2
import argparse
import subprocess


def get_all_files(directory):
    files = glob.glob(directory, recursive=True)
    return files


img_h, img_w = 540, 960
camera_img_w = img_w * 3 // 4  # downscale 0.25


def split_and_save(img_file, output_path):
    img_name = img_file.split('/')[-1]
    image = cv2.imread(img_file)

    height, width, _ = image.shape

    left_width = int((width - camera_img_w) // 2)

    left_x_start = 0
    left_x_end = left_width

    right_x_start = left_width + camera_img_w
    right_x_end = width

    # crop bev gt and generated bev
    left_image = image[:, left_x_start:left_x_end]
    right_image = image[:, right_x_start:right_x_end]

    # save
    real_img_path = f'{output_path}/real/'
    fake_img_path = f'{output_path}/fake/'
    os.makedirs(real_img_path, exist_ok=True)
    os.makedirs(fake_img_path, exist_ok=True)
    cv2.imwrite(real_img_path + img_name, left_image)
    cv2.imwrite(fake_img_path + img_name, right_image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='old foo help')
    args = parser.parse_args()

    input_path = f"{args.folder}/eval/image/**/*.png"
    output_path = f"{args.folder}/eval"

    fid_k = 2000
    all_files = get_all_files(input_path)
    # split eval image into real and fake
    for img_file in tqdm(all_files[:fid_k]):
        split_and_save(img_file, output_path)

    command = ["python", "-m", "pytorch_fid", f"{output_path}/real", f"{output_path}/fake"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
    except Exception as e:
        print("An error occurred:", e)
