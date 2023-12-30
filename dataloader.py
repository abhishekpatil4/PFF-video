import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class FrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_files = glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True)
        self.classes = sorted({os.path.basename(os.path.dirname(f)) for f in self.frame_files})

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.frame_files[idx]
        image = Image.open(img_name)
        class_name = os.path.basename(os.path.dirname(img_name))
        class_index = self.classes.index(class_name)

        if self.transform:
            image = self.transform(image)

        return image, class_index
    
