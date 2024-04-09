import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image 
from io import BytesIO
from scipy.ndimage import gaussian_filter
from model import CLIPViTL14Model
import seaborn as sns
import matplotlib.pyplot as plt



MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)


def plot_pie_chart(false_prob, save_path):
    labels = ['Real', 'Fake']
    probabilities = [1-false_prob, false_prob]
    colors = ['#ADD8E6', '#FFC0CB']  # 浅蓝色和浅红色
    explode = (0.1, 0)  # 设置偏移量

    plt.figure(figsize=(6, 6))
    plt.pie(probabilities, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.savefig(save_path)


def detect(
    img_path: str,
    save_path: str,
    pretrained_path: str=None,
    stat_from: str="clip",
    gaussian_sigma: int=None,
    jpeg_quality: int=None,
    device: str="cpu"
):
    img = Image.open(img_path).convert("RGB")
    if gaussian_sigma is not None:
        img = gaussian_blur(img, gaussian_sigma) 
    if jpeg_quality is not None:
        img = png2jpg(img, jpeg_quality)
    
    # transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
    ])
    img = transform(img)
    img: torch.Tensor
    if img.ndim == 3:
        img = img.unsqueeze(dim=0)
    img = img.to(device=device)
    model = CLIPViTL14Model()
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.fc.load_state_dict(state_dict)
        model.eval()
        model.to(device=device)
    probs = model(img).sigmoid().flatten().tolist()[0]
    plot_pie_chart(probs, save_path)