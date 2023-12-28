from PIL import Image
import numpy as np

import torch
import torch.nn
import torch.nn as nn
import torchvision
from tqdm import tqdm
from dataset import ImageDataset, LargeImageDataset
import torch.optim as optim
from torchvision import transforms, models


import INR
import utils
import clip
import torch.nn.functional as F

from PIL import Image 
import PIL 
from torchvision import utils as vutils
import argparse
from criteria.perceptual_loss import VGGPerceptualLoss
import cv2
from clipseg_models.clipseg import CLIPDensePredT
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import os
import random
from template import imagenet_templates


parser = argparse.ArgumentParser()

parser.add_argument('--content_path', type=str, default="./test_set/church.jpeg",
                    help='Image resolution')
parser.add_argument('--content_name', type=str, default="church",
                    help='Image resolution')
parser.add_argument('--text', type=str, default="wood",
                    help='Image resolution')
parser.add_argument('--save_path', type=str, default="results_output/test",
                    help='Image resolution')
parser.add_argument('--lambda_frontreg', type=float, default=0.2,
                    help='foreground regularization loss parameter')
parser.add_argument('--lambda_tv', type=float, default=0.000002,
                    help='total variation loss parameter')
parser.add_argument('--lambda_patch', type=float, default=35,
                    help='PatchCLIP loss parameter')
parser.add_argument('--lambda_c', type=float, default=2,
                    help='content loss parameter')
parser.add_argument('--num_crops', type=int, default=64*2,
                    help='number of patches')
parser.add_argument('--img_width', type=int, default=512,
                    help='size of images')
parser.add_argument('--img_height', type=int, default=512,
                    help='size of images')
parser.add_argument('--max_step', type=int, default=200,
                    help='Number of domains')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VGG = models.vgg19(pretrained=True).features
VGG.to(device)

os.makedirs(os.path.join(args.save_path), exist_ok=True)

for parameter in VGG.parameters():
    parameter.requires_grad_(False)
    
def img_denormalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = image*std +mean
    return image

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

    
def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

content_path = args.content_path
content_image = utils.load_image2(content_path, img_height=args.img_height,img_width=args.img_width)
content = args.content_name

content_image = content_image.to(device)

seg_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
seg_model.eval();

# non-strict, because we only stored decoder weights (not CLIP weights)
seg_model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device('cuda')), strict=False);


prompts = content

photo_name = content_path.split('/')[-1].split('.')[0]
# predict
with torch.no_grad():
    mask_ = seg_model(img_normalize(content_image), prompts)[0]
    mask = F.relu(torch.sigmoid(mask_)-0.4)
    mask = torch.where(mask>0, mask+0.2, mask)
    blur = torchvision.transforms.GaussianBlur(kernel_size=(101,101), sigma=(1.4, 10.0))
    mask = blur(mask)
    # mask = (mask > (mask.max()/2)).float()
    inv_mask = 1 - mask

binary_mask = (mask > 0.4).float()
inv_binary_mask = 1 - binary_mask
coordinates = np.argwhere(binary_mask)

# out_path = os.path.join(args.save_path, photo_name +'_'+args.content_name+'_mask.jpg')
# vutils.save_image(
#                 mask,
#                 out_path,
#                 nrow=1,
#                 normalize=True)


input_image = Image.open(content_path).convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((args.img_height, args.img_width)),
])

img = transform(input_image).unsqueeze(0)
input_image = input_image.resize((args.img_height, args.img_width))

heatmap = torch.sigmoid(mask_[0][0]).cpu().detach().numpy()
heatmap = gray2rgb(heatmap*255)
img2 = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
super_imposed_img = cv2.addWeighted(img2, 0.5, np.asarray(input_image).astype(np.uint8), 0.5, 0)
out_path = os.path.join(args.save_path, photo_name +'_'+args.content_name+'_heatmap.jpg')
cv2.imwrite(out_path, super_imposed_img)

mask_cpu = mask.squeeze(0).squeeze(0).cpu().detach().numpy()
input = Image.open(content_path).convert('RGB')
input = input.resize((args.img_height, args.img_width))
content_features = utils.get_features(img_normalize(content_image*mask.to(device)), VGG)

target = content_image.clone().requires_grad_(True).to(device)

network_size = (8, 512, 256)
mapping_size = 256  

B_gauss = torch.randn((mapping_size, 2)).to(device) * 10

ds = ImageDataset(args.content_path, 512)

grid, image = ds[0]
grid = grid.unsqueeze(0).to(device)
image = image.unsqueeze(0).to(device)

im = cv2.cvtColor(cv2.imread(args.content_path), cv2.COLOR_BGR2RGB)
h,w,c = im.shape

if h>=1200 and h<1700:
    h,w = h//2, w//2

elif h>=1700:
    h,w = h//3, w//3

elif h<600:
    h,w = h*2, w*2

dt = LargeImageDataset(args.content_path, int(w),int(h))

grid_test, image_test = dt[0]
grid_test = grid_test.unsqueeze(0).to(device)
image_test = image_test.unsqueeze(0).to(device)

test_data = (grid_test, image_test)
train_data = (grid, image)

model = INR.gon_model(*network_size).to(device)

loss_fn = torch.nn.MSELoss()

content_weight = args.lambda_c

show_every = 100
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
steps = args.max_step

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []

output_image = content_image

m_cont = torch.mean(content_image,dim=(2,3),keepdim=False).squeeze(0)
m_cont = [m_cont[0].item(),m_cont[1].item(),m_cont[2].item()]

augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])

def randomcrop(img, size):

    random_cropper = transforms.Compose([
        transforms.RandomCrop(size)
    ])

    return random_cropper(img)


resized = transforms.Compose([
    
    transforms.Resize(512)
])
device='cuda'
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

prompt = args.text

source = "a Photo"

with torch.no_grad():
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
    source_features = clip_model.encode_image(clip_normalize(content_image,device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
    
mask = mask.to(device)
inv_mask = inv_mask.to(device)

# print(content_image.shape)
front_content = (content_image * mask)
back_content = (content_image * inv_mask)
import time
start = time.time()
num_crops = args.num_crops
perceptual_loss = VGGPerceptualLoss().cuda()

model_input = INR.input_mapping(train_data[0], B_gauss)
test_input = INR.input_mapping(test_data[0], B_gauss)
pbar = tqdm(range(0,steps+1))

for epoch in pbar:
    
    model.train()

    scheduler.step()
    target = model(model_input).permute(0,3,1,2) #+ content_image
    target.requires_grad_(True)

    front_target = (target * mask)
    back_target = (content_image * inv_mask)
    out_img = front_target + back_target

    target_features = utils.get_features(img_normalize(front_target), VGG)
    
    content_loss = 0

    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    loss_patch=0 
    img_proc =[]

    for n in range(num_crops):
        idx = np.random.randint(0, len(coordinates[0]))
        randnum = random.randint(64,256)
        y_position, x_position = coordinates[2][idx], coordinates[3][idx]
        target_crop = transforms.functional.crop(out_img*mask, y_position-int(randnum/2), x_position-int(randnum/2), randnum, randnum)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)

    img_proc = torch.cat(img_proc,dim=0)
    img_aug = img_proc

    image_features = clip_model.encode_image(clip_normalize(img_aug,device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    
    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
    
    text_direction = (text_features-text_source).repeat(image_features.size(0),1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))

    loss_temp = loss_temp.sort(descending=False, stable=True)
    loss_temp = loss_temp.values[:int(num_crops*0.5)]
    loss_patch+=loss_temp.mean()
    
    reg_tv = get_image_prior_losses(back_target)
    front_pl = perceptual_loss(out_img, content_image)
    total_loss =  args.lambda_patch * loss_patch + args.lambda_c * content_loss  + args.lambda_frontreg*front_pl +  args.lambda_tv * reg_tv 
    total_loss_epoch.append(total_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    pbar.set_postfix({'loss': total_loss.item()})

    if epoch %20 ==0:
        out_path = os.path.join(args.save_path, photo_name +'_'+args.content_name+'_'+prompt+'.jpg')
        # out_path = './outputs/'+prompt+'_'+content+'_'+exp+'.jpg'
        output_image = out_img.clone() 
        output_image = torch.clamp(output_image,0,1)
        # output_image = adjust_contrast(output_image,1.5)
        vutils.save_image(
                                    output_image,
                                    out_path,
                                    nrow=1,
                                    normalize=True)
        
        model.eval()
        with torch.no_grad():
            out_path = os.path.join(args.save_path, photo_name +'_'+args.content_name+'_'+prompt+'large_test.jpg')
            test_image = model(test_input).permute(0,3,1,2)* F.interpolate(mask, size=(int(h), int(w)))+ F.interpolate(image_test.permute(0,3,1,2), size=(int(h), int(w))) * F.interpolate(inv_mask, size=(int(h), int(w)))
            # test_image = adjust_contrast(test_image,1.5)
            torchvision.utils.save_image(test_image, out_path)
        


end = time.time()
print(end-start)
