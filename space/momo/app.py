import time
import shutil
import gradio as gr
import os
import cv2
import json
import torch
import argparse
import numpy as np
from lib.data import get_meanpose
from lib.network import get_autoencoder
from lib.util.motion import preprocess_mixamo, preprocess_test, postprocess
from lib.util.general import get_config
from lib.operation import rotate_and_maybe_project_world
from itertools import combinations
from lib.util.visualization import motion2video
from PIL import Image

def load_and_preprocess(path, config, mean_pose, std_pose):

    motion3d = np.load(path)

    # length must be multiples of 8 due to the size of convolution
    _, _, T = motion3d.shape
    T = (T // 8) * 8
    motion3d = motion3d[:, :, :T]

    # project to 2d
    motion_proj = motion3d[:, [0, 2], :]

    # reformat for mixamo data
    motion_proj = preprocess_mixamo(motion_proj, unit=1.0)

    # preprocess for network input
    motion_proj, start = preprocess_test(motion_proj, mean_pose, std_pose, config.data.unit)
    motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))
    motion_proj = torch.from_numpy(motion_proj).float()

    return motion_proj, start

def image2video():
        # 指定包含PNG图像的目录
    image_directory = './an-frames'
    # 输出视频的路径
    output_video_path = './adv.mp4'
    # 视频的帧率（每秒中的帧数）
    fps = 24

    # 获取所有PNG图像的文件名并按顺序排序
    images = sorted([img for img in os.listdir(image_directory) if img.endswith(".png")])

    # 确定视频的分辨率
    first_image_path = os.path.join(image_directory, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape
    size = (width, height)

    # 创建视频写入器
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    # 依次读取图像并写入视频
    for img_name in images:
        img_path = os.path.join(image_directory, img_name)
        frame = cv2.imread(img_path)
        
        # 确保帧的大小与视频的分辨率匹配
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        
        out.write(frame)

    # 关闭视频写入器
    out.release()

    print(f"Video saved to {output_video_path}")

def handle_motion_generation(npy1,npy2):
    path1 = './data/a.npy'
    path2 = './data/b.npy'
    np.save(path1,npy1)
    np.save(path2,npy2)
    config_path = './configs/transmomo.yaml'  # 替换为您的配置文件路径
    description_path = "./data/mse_description.json"
    checkpoint_path = './data/autoencoder_00200000.pt'  
    out_dir_path = './output'  # 替换为输出目录的路径

    config = get_config(config_path)
    ae = get_autoencoder(config)
    ae.load_state_dict(torch.load(checkpoint_path))
    # ae.cuda()
    ae.eval()
    mean_pose, std_pose = get_meanpose("test", config.data)
    # print("loaded model")

    description = json.load(open(description_path))
    chars = list(description.keys())

    os.makedirs(out_dir_path, exist_ok=True)

    path1 = './data/1.npy'
    path2 = './data/2.npy'
    out_path1 = os.path.join(out_dir_path, "adv.npy")


    x_a, x_a_start = load_and_preprocess(path1, config, mean_pose, std_pose)
    x_b, x_b_start = load_and_preprocess(path2, config, mean_pose, std_pose)
    
    # x_a_batch = x_a.unsqueeze(0).cuda()
    # x_b_batch = x_b.unsqueeze(0).cuda()
    x_a_batch = x_a.unsqueeze(0)
    x_b_batch = x_b.unsqueeze(0)

    x_ab = ae.cross2d(x_a_batch, x_b_batch, x_a_batch)
    x_ab = postprocess(x_ab, mean_pose, std_pose, config.data.unit, start=x_a_start)

    np.save(out_path1, x_ab)
    motion_data = x_ab
    height = 512  # 视频的高度
    width = 512   # 视频的宽度
    save_path = './an.mp4'  # 保存视频的路径
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 关节颜色
    bg_color = (255, 255, 255)  # 背景颜色
    fps = 25  # 视频的帧率

    # print(motion_data.shape)
    # 调用函数生成视频
    motion2video(motion_data, height, width, save_path, colors, bg_color=bg_color, transparency=False, fps=fps)
    image2video()
    first_frame_image = Image.open('./an-frames/0000.png')
    return './adv.mp4'

# 创建 Gradio 界面并展示视频而不是图片
with gr.Blocks() as demo:
    gr.Markdown("Upload two `.npy` files to generate motion and visualize the animation.")

    with gr.Row():
        file1 = gr.File(file_types=[".npy"], label="Upload first .npy file")
        file2 = gr.File(file_types=[".npy"], label="Upload second .npy file")
    
    with gr.Row():
        generate_btn = gr.Button("Generate Animation")
    
    output_video = gr.Video(label="Generated Animation",width = 500)

    generate_btn.click(
        fn=handle_motion_generation,
        inputs=[file1, file2],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch()
