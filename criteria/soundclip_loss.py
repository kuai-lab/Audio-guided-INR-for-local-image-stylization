import torch
import clip
import torch 
from collections import OrderedDict
import math
import timm
import torch.nn as nn
import numpy as np

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class SwinAudioEncoder(torch.nn.Module):

    def __init__(self):

        super(SwinAudioEncoder, self).__init__()

        self.feature_extractor = timm.create_model("swin_tiny_patch4_window7_224", num_classes=512, pretrained=True, in_chans=1)
        self.logit_scale_ai = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_at = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        h = self.feature_extractor(x)
        return h


class SoundCLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(SoundCLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=512 // 32)

        self.audio_encoder = SwinAudioEncoder()
        self.audio_encoder.load_state_dict(copyStateDict(torch.load("./weights/swin_audioencoder.pth")))

        self.audio_encoder = self.audio_encoder.cuda()
        self.audio_encoder.eval()

    def forward(self, audio):
        audio_features = self.audio_encoder(audio).float()
        return audio_features


