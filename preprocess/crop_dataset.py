#!/usr/bin/env python

import numpy as np
import cv2
import os

from random import shuffle
from tqdm import tqdm

train_dir = 'instrument_dataset_9'


dirs = os.listdir(train_dir)
for folders in dirs:
    if folders == 'images':
        req_train_dir = os.path.join(train_dir,folders)
        for img_pt in tqdm(os.listdir(req_train_dir)):
            path = os.path.join(req_train_dir,img_pt)
            #print(path)
            img = cv2.imread(path,1)
            req_img = img[28:(1024+28),320:(1280+320)]
            cv2.imwrite(path,req_img)
