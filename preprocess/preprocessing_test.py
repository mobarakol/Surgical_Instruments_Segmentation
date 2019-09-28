from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np

data_path = Path('Test_Orig')

train_path = data_path

cropped_train_path = data_path / 'cropped_test'

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

if __name__ == '__main__':
    for instrument_index in range(1, 11):
        instrument_folder = 'instrument_dataset_' + str(instrument_index)
        (cropped_train_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)
        binary_mask_folder = (cropped_train_path / instrument_folder / 'binary_masks')
        binary_mask_folder.mkdir(exist_ok=True, parents=True)

        parts_mask_folder = (cropped_train_path / instrument_folder / 'parts_masks')
        parts_mask_folder.mkdir(exist_ok=True, parents=True)

        instrument_mask_folder = (cropped_train_path / instrument_folder / 'instruments_masks')
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)
        mask_folders_binary = (train_path / instrument_folder / 'BinarySegmentation')
        mask_folders_parts = (train_path / instrument_folder / 'PartsSegmentation')
        mask_folders_type = (train_path / instrument_folder / 'TypeSegmentation')
        #print(mask_folders_type)
        for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*'))):
            #print(file_name.name)
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape
            img = img[h_start: h_start + height, w_start: w_start + width]
            cv2.imwrite(str(cropped_train_path / instrument_folder / 'images' / (file_name.stem + '.jpg')), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            
            
            img_binary = cv2.imread(str(mask_folders_binary / file_name.name), 0)
            #print(np.unique(img_binary))
            img_binary[img_binary>0] = 1
            img_binary = img_binary[h_start: h_start + height, w_start: w_start + width]
            cv2.imwrite(str(cropped_train_path / instrument_folder / 'binary_masks' / (file_name.stem + '.png')), img_binary,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            img_parts = cv2.imread(str(mask_folders_parts / file_name.name), 0)
            #print(np.unique(img_parts))
            img_parts[img_parts == 30] = 1  # Shaft
            img_parts[img_parts == 100] = 2  # Wrist
            img_parts[img_parts == 255] = 3  # Claspers
            #print(np.unique(img_parts))
            img_parts = img_parts[h_start: h_start + height, w_start: w_start + width]
            cv2.imwrite(str(cropped_train_path / instrument_folder / 'parts_masks' / (file_name.stem + '.png')), img_parts,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            #print(file_name.name)
            #print(mask_folders_type / file_name.name)
            img_type = cv2.imread(str(mask_folders_type / file_name.name), 0)
            #print(np.unique(img_type))
            img_type = img_type[h_start: h_start + height, w_start: w_start + width]
            cv2.imwrite(str(cropped_train_path / instrument_folder / 'instruments_masks' / (file_name.stem + '.png')), img_type,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

