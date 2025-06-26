import os
import cv2

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    while frame_number < frame_count:
        ret, frame = video.read()

        if not ret:
            break

        output_path = os.path.join(output_folder, f"{frame_number:08d}.png")
        cv2.imwrite(output_path, frame)
        frame_number += 1

    video.release()
    return frame_number

input_folder = "/data/home/yifanfliu/dataset/AdcSR/384"
output_folder = "/data/home/yifanfliu/dataset/AdcSR/384_frame"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
subfolder_count = 1

for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    # subfolder_name = f"{subfolder_count:05d}"
    subfolder_name = video_file.split('.')[0]
    subfolder_path = os.path.join(output_folder, subfolder_name)

    frame_count = extract_frames(video_path, subfolder_path)
    print(f"from {video_file} extracting {frame_count} frame, stored in {subfolder_name}")

    subfolder_count += 1
