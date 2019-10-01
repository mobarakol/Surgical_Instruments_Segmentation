#System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F
from instruments_data2017.instruments_data import instruDataset
from model import InstrumentsMFF

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
args = {
    'num_class': 8,
    'num_gpus': 2,
    'num_epoch': 200,
    'batch_size': 6,
    'lr': 0.0001,
    'lr_decay': 0.9,
    'w_decay': 1e-4,
    'ckpt_dir': 'ckpt/instruments_type/'
}

if not os.path.exists(args['ckpt_dir']):
    os.makedirs(args['ckpt_dir'])

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

if __name__ == '__main__':
    #img_dir = 'instruments_data2017/train_mine.txt'
    #img_dir = '/media/mmlab/data/Datasets/Instruments/2017/train_mine.txt'
    img_dir = '/media/mobarak/data/Datasets/Instruments/2017/train_mine.txt'
    dataset = instruDataset(img_dir=img_dir)
    train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2,
                              drop_last=True)
    model = InstrumentsMFF(n_classes=args['num_class']).cuda()
    model = torch.nn.parallel.DataParallel(model, device_ids=range(args['num_gpus']))
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['w_decay'])
    criterion = CrossEntropyLoss2d(size_average=True).cuda()
    model.train()
    epoch_iters = dataset.__len__() / args['batch_size']
    for epoch in range(args['num_epoch']):
        for batch_idx, data in enumerate(train_loader):
            inputs, labels, labels_aux = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            labels_aux = Variable(labels_aux).cuda()
            optimizer.zero_grad()
            outputs, outputs_aux = model(inputs)
            main_loss = criterion(outputs, labels)
            aux_loss = criterion(outputs_aux, labels_aux)
            loss = main_loss + 0.4 *aux_loss
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 50 == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.10f]' % (
                    epoch, batch_idx + 1, epoch_iters, loss.item(),
                    optimizer.param_groups[0]['lr']))

        snapshot_name = 'epoch_' + str(epoch)
        torch.save(model.state_dict(), os.path.join(args['ckpt_dir'], snapshot_name + '.pth.tar'))