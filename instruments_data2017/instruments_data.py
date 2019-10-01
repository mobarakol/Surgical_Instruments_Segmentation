from PIL import Image
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class instruDataset(Dataset):
    def __init__(self, img_dir, is_train=None):
        self.is_train = is_train
        idx = 0
        file_img = open(img_dir, 'r')
        self.img_anno_pairs = {}
        for line in file_img:
            self.img_anno_pairs[idx] = line[0:-1]
            idx = idx + 1

    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        _img = Image.open(self.img_anno_pairs[index] +'.jpg').convert('RGB')
        _target = Image.open(self.img_anno_pairs[index][:-15] + 'instruments_masks/'
                             + os.path.basename(self.img_anno_pairs[index]) + '.png')
        # _img = Image.open(self.img_anno_pairs[index][:-26] + 'images/' + os.path.basename(
        #     self.img_anno_pairs[index]) + '.jpg').convert('RGB')
        # _target = Image.open(self.img_anno_pairs[index] + '.png')
        if self.is_train:
            hflip = random.random() < 0.5
            if hflip:
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                _target = _target.transpose(Image.FLIP_LEFT_RIGHT)

        _img = torch.from_numpy(np.array(_img).transpose(2,0,1)).float()
        _target_main = torch.from_numpy(np.array(_target)).long()
        _target_aux = _target.resize((160, 128), Image.NEAREST)
        _target_aux = torch.from_numpy(np.array(_target_aux)).long()
        return _img, _target_main, _target_aux