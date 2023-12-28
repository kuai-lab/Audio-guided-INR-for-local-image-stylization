import cv2
import torch
from torch.utils.data import Dataset


def create_grid(h, w, device="cpu"):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


class ImageDataset(Dataset):

    def __init__(self, image_path, img_dim):
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        left_w = int((w - h) / 2)

        image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255

        grid = create_grid(*self.img_dim[::-1])

        return grid, torch.tensor(image, dtype=torch.float32)

    def __len__(self):
        return 1
        
class LargeImageDataset(Dataset):

    def __init__(self, image_path, w, h):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.w = w
        self.h = h
        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255

        grid = create_grid(self.h, self.w)
        return grid, torch.tensor(image, dtype=torch.float32)

    def __len__(self):
        return 1


