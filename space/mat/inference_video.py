import argparse
import cv2
import torch
import os
import shutil

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm
from PIL import Image
import gradio as gr
from dataset import VideoDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment


# --------------- Arguments ---------------



# --------------- Utils ---------------


class VideoWriter:
    def __init__(self, path, frame_rate, width, height):
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        
    def add_batch(self, frames):
        frames = frames.mul(255).byte()
        frames = frames.cpu().permute(0, 2, 3, 1).numpy()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame)
            

class ImageSequenceWriter:
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path)
        
    def add_batch(self, frames):
        Thread(target=self._add_batch, args=(frames, self.index)).start()
        self.index += frames.shape[0]
            
    def _add_batch(self, frames, index):
        frames = frames.cpu()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = to_pil_image(frame)
            frame.save(os.path.join(self.path, str(index + i).zfill(5) + '.' + self.extension))


# --------------- Main ---------------

def video_matting(video_src_content,video_bgr_content):
    src_video_path = './source/src_video.mp4'
    bgr_image_path = './source/bgr_image.png'
    with open(src_video_path, 'wb') as video_file:
        video_file.write(video_src_content)
    
    # 写入背景图片文件
    with open(bgr_image_path, 'wb') as bgr_file:
        bgr_file.write(video_bgr_content)
    video_src = src_video_path
    video_bgr = bgr_image_path
    default_args = {
    'model_type': 'mattingrefine',
    'model_backbone': 'resnet50',
    'model_backbone_scale': 0.25,
    'model_refine_mode': 'sampling',
    'model_refine_sample_pixels': 80000,
    'model_checkpoint': './pytorch_resnet50.pth',  
    'model_refine_threshold':0.7,
    'model_refine_kernel_size':3,
    'video_src': './source/src.mp4',          
    'video_bgr': './source/bgr.png',          
    'video_target_bgr': None,
    'video_resize': [1920, 1080],
    'device': 'cpu',  # 默认设置为CPU
    'preprocess_alignment': False,
    'output_dir': './output',        
    'output_types': ['com'],
    'output_format': 'video'
    }

    args = argparse.Namespace(**default_args)
    device = torch.device(args.device)

    # Load model
    if args.model_type == 'mattingbase':
        model = MattingBase(args.model_backbone)
    if args.model_type == 'mattingrefine':
        model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)


    # Load video and background
    vid = VideoDataset(video_src)
    bgr = [Image.open(video_bgr).convert('RGB')]
    dataset = ZipDataset([vid, bgr], transforms=A.PairCompose([
        A.PairApply(T.Resize(args.video_resize[::-1]) if args.video_resize else nn.Identity()),
        HomographicAlignment() if args.preprocess_alignment else A.PairApply(nn.Identity()),
        A.PairApply(T.ToTensor())
    ]))
    if args.video_target_bgr:
        dataset = ZipDataset([dataset, VideoDataset(args.video_target_bgr, transforms=T.ToTensor())])

    # Create output directory
    # if os.path.exists(args.output_dir):
    #     if input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
    #         shutil.rmtree(args.output_dir)
    #     else:
    #         exit()
    # os.makedirs(args.output_dir)


    # Prepare writers
    if args.output_format == 'video':
        h = args.video_resize[1] if args.video_resize is not None else vid.height
        w = args.video_resize[0] if args.video_resize is not None else vid.width
        if 'com' in args.output_types:
            com_writer = VideoWriter(os.path.join(args.output_dir, 'com.mp4'), vid.frame_rate, w, h)
        if 'pha' in args.output_types:
            pha_writer = VideoWriter(os.path.join(args.output_dir, 'pha.mp4'), vid.frame_rate, w, h)
        if 'fgr' in args.output_types:
            fgr_writer = VideoWriter(os.path.join(args.output_dir, 'fgr.mp4'), vid.frame_rate, w, h)
        if 'err' in args.output_types:
            err_writer = VideoWriter(os.path.join(args.output_dir, 'err.mp4'), vid.frame_rate, w, h)
        if 'ref' in args.output_types:
            ref_writer = VideoWriter(os.path.join(args.output_dir, 'ref.mp4'), vid.frame_rate, w, h)
    else:
        if 'com' in args.output_types:
            com_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'com'), 'png')
        if 'pha' in args.output_types:
            pha_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'pha'), 'jpg')
        if 'fgr' in args.output_types:
            fgr_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'fgr'), 'jpg')
        if 'err' in args.output_types:
            err_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'err'), 'jpg')
        if 'ref' in args.output_types:
            ref_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'ref'), 'jpg')
        

    # Conversion loop
    with torch.no_grad():
        for input_batch in tqdm(DataLoader(dataset, batch_size=1, pin_memory=True)):
            if args.video_target_bgr:
                (src, bgr), tgt_bgr = input_batch
                tgt_bgr = tgt_bgr.to(device, non_blocking=True)
            else:
                src, bgr = input_batch
                tgt_bgr = torch.tensor([120/255, 255/255, 155/255], device=device).view(1, 3, 1, 1)
            src = src.to(device, non_blocking=True)
            bgr = bgr.to(device, non_blocking=True)
            
            if args.model_type == 'mattingbase':
                pha, fgr, err, _ = model(src, bgr)
            elif args.model_type == 'mattingrefine':
                pha, fgr, _, _, err, ref = model(src, bgr)
            elif args.model_type == 'mattingbm':
                pha, fgr = model(src, bgr)

            if 'com' in args.output_types:
                if args.output_format == 'video':
                    # Output composite with green background
                    com = fgr * pha + tgt_bgr * (1 - pha)
                    com_writer.add_batch(com)
                else:
                    # Output composite as rgba png images
                    com = torch.cat([fgr * pha.ne(0), pha], dim=1)
                    com_writer.add_batch(com)
            if 'pha' in args.output_types:
                pha_writer.add_batch(pha)
            if 'fgr' in args.output_types:
                fgr_writer.add_batch(fgr)
            if 'err' in args.output_types:
                err_writer.add_batch(F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False))
            if 'ref' in args.output_types:
                ref_writer.add_batch(F.interpolate(ref, src.shape[2:], mode='nearest'))

    return './output/com.mp4'

# 读取本地视频文件的二进制数据
def get_video_content(video_path):
    with open(video_path, 'rb') as file:
        video_content = file.read()
    return video_content

# 假设你的视频文件路径是'./local_video.mp4'
local_video_path = './output/com.mp4'
local_video_content = get_video_content(local_video_path)

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## Video Matting")
    with gr.Row():
        video_src = gr.File(label="Upload Source Video (.mp4)", type="binary", file_types=["mp4"])
        video_bgr = gr.File(label="Upload Background Image (.png)", type="binary", file_types=["png"])
    with gr.Row():
        output_video = gr.Video(label="Result Video")
    submit_button = gr.Button("Start Matting")
    
    # def download_video(video_path):
    #     if os.path.exists(video_path):
    #         with open(video_path, 'rb') as file:
    #             video_data = file.read()
    #         return video_data, "video/mp4", os.path.basename(video_path)
    #     else:
    #         return "Not Found", "text/plain", None

    def clear_outputs():
        output_video.update(value=None)
    
    submit_button.click(
        fn=video_matting,
        inputs=[video_src, video_bgr],
        outputs=[output_video]
    )
    # download_button = gr.Button("Download")
    # download_button.click(
    #     download_video,
    #     inputs=[output_video],  # 从视频组件传递视频路径
    #     outputs=[gr.File(label="Download")]
    # )
    clear_button = gr.Button("Clear")
    clear_button.click(fn=clear_outputs, inputs=[], outputs=[])

if __name__ == "__main__":
    demo.launch()