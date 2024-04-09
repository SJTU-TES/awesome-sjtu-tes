from clip.clip import load
import torch.nn as nn


class CLIPViTL14Model(nn.Module):
    def __init__(self, num_classes=1):
        super(CLIPViTL14Model, self).__init__()
        self.model, self.preprocess = load("ViT-L/14", device="cpu")
        self.fc = nn.Linear(768, num_classes)
 
    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)