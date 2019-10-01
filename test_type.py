#System
import numpy as np
import math
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import time
from scipy.spatial.distance import directed_hausdorff
from numpy.core.umath_tests import inner1d
from skimage.io import imread, imsave
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
from model import InstrumentsMFF
from torchsummary import summary
from instruments_data2017.instruments_data import instruDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

args = {
    'exp_name': 'instruments-instru',
    'snapshot': '',
    'num_class': 8,
    'batch_size':1,
    'num_gpus':1,
    'ckpt_dir': 'ckpt/instruments-instru/',
}
def HausdorffDist(A,B):
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
    return (dH)

def dice(pred, label):
    dice_val = np.float(np.sum(pred[label == 1] == 1)) * 2.0 / (np.float(np.sum(label == 1) + np.sum(pred == 1)));
    return dice_val

def specificity(TP, TN, FP, FN):
    return TN / (FP + TN)


def sensitivity(TP, TN, FP, FN):
    return TP / (TP + FN)

def spec_sens(pred, gt):
    # pred[pred>0] = 1
    # gt[gt>0] = 1
    A = np.logical_and(pred, gt)
    TP = float(A[A > 0].shape[0])
    TN = float(A[A == 0].shape[0])
    B = img_pred - labels
    FP = float(B[B > 0].shape[0])
    FN = float(B[B < 0].shape[0])
    specificity = TN / (FP + TN)
    sensitivity = TP / (TP + FN)
    return specificity, sensitivity


if __name__ == '__main__':
    #img_dir = '/media/mmlab/data/Datasets/Instruments/2017/test_mine.txt'
    img_dir = '/media/mobarak/data/Datasets/Instruments/2017/test_mine.txt'
    dataset = instruDataset(img_dir=img_dir, is_train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2)
    model = InstrumentsMFF(n_classes=args['num_class'])
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    # summary(model, (3, 1024, 1280))
    print(dataset.__len__())
    Best_Dice = 0
    Best_epoch=0

    for epochs in range(76, 77):
        args['snapshot'] = 'epoch_' + str(epochs) + '.pth.tar'
        #model.load_state_dict(torch.load(os.path.join(args['ckpt_dir'],args['snapshot']))) # test all ckpts
        model.load_state_dict(torch.load(os.path.join(args['snapshot']))) # test on final trained model
        model.eval()
        w, h = 0, args['num_class']
        mdice = [[0 for x in range(w)] for y in range(h)]
        mspecificity = [[0 for x in range(w)] for y in range(h)]
        msensitivity = [[0 for x in range(w)] for y in range(h)]
        mhausdorff = [[0 for x in range(w)] for y in range(h)]
        haus = []
        mytime = []
        cc = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                inputs, labels,mpath = data
                inputs = Variable(inputs).cuda()
                t0 = time.time()
                outputs = model(inputs)
                t1 = time.time()
                mytime.append((t1 - t0))
                img_pred = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
                labels = np.array(labels)
                # labels[labels>0] = 1
                # img_pred[img_pred > 0] = 1
                for dice_idx in range(0,img_pred.shape[0]):
                    cc = cc + 1
                    if(np.max(labels[dice_idx])==0):
                        continue
                    labs = np.unique(labels[dice_idx])
                    for instru_idx in range(1, len(labs)):
                        labels_temp = np.zeros(labels.shape[1:])
                        img_pred_temp = np.zeros(labels.shape[1:])
                        labels_temp[labels[dice_idx] == labs[instru_idx]] = 1
                        img_pred_temp[img_pred[dice_idx] == labs[instru_idx]] = 1
                        if (np.max(labels_temp) == 0):# or (np.max(img_pred_temp)==0):
                            continue
                        mdice[labs[instru_idx]].append(dice(img_pred_temp, labels_temp))
                        # mhausdorff[labs[instru_idx]].append(directed_hausdorff(img_pred_temp, labels_temp)[0])
                        # spec, sens = spec_sens(img_pred_temp, labels_temp)
                        # mspecificity[labs[instru_idx]].append(spec)
                        # msensitivity[labs[instru_idx]].append(sens)



        avg_dice = []
        avg_hd = []
        avg_spec = []
        avg_sens = []
        for idx_eval in range(1, args['num_class']):
            if idx_eval == 2 or idx_eval == 3 or idx_eval == 7:# or math.isnan(float(np.mean(mdice[idx_eval]))):
            # if idx_eval == 3 or idx_eval == 4 or idx_eval == 5 or idx_eval == 6 or math.isnan(
            #             float(np.mean(mdice[idx_eval]))):
                mdice[idx_eval] = 0
                continue
            avg_dice.append(np.mean(mdice[idx_eval]))
            # avg_hd.append(np.mean(mhausdorff[idx_eval]))
            # avg_spec.append(np.mean(mspecificity[idx_eval]))
            # avg_sens.append(np.mean(msensitivity[idx_eval]))

        if np.mean(avg_dice) > Best_Dice:
            Best_Dice = np.mean(avg_dice)
            Best_epoch = epochs

        print(str(cc), str(epochs) + ' Mean Dice:' + str(np.mean(avg_dice)) +' Each:'+ str(avg_dice) +'   Best='+str(Best_epoch)+':'+str(Best_Dice))


        # print(str(epochs), ' Mean Dice:', str(np.mean(avg_dice)), ' Each:', str(avg_dice),'\n',
        #       ' Mean Hausdorff:', str(np.mean(avg_hd)), ' Each:', str(avg_hd), '\n',
        #       ' Mean Specificity:', str(np.mean(avg_spec)), ' Each:', str(avg_spec), '\n',
        #       ' Mean Sensitivity:', str(np.mean(avg_sens)), ' Each:', str(avg_sens), '\n',
        #       'Avg Time(ms):', np.mean(mytime) * 1000, 'fps:', (1.0 / np.mean(mytime)))