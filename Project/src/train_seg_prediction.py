### to calculate total parameters in pytorch model
### sum(p.numel() for p in model.parameters())


from __future__ import print_function
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR
import torch.nn as nn
import time
from math import log10
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')
from os.path import join
import json
import argparse
from shutil import copyfile
from modeling.deeplab import *
from cityscape_dataloader.cityscapes import Cityscapes
from cityscape_dataloader.custom_transforms import Project, DepthConversion, PIL_To_Tensor, NormalizeRange
from torchvision import transforms
from sklearn.metrics import jaccard_similarity_score as jsc
from custom_callbacks.Loss_plotter import LossPlotter
from custom_callbacks.Logger import Logger
import pandas as pd
import PIL.Image as Image

torch.cuda.manual_seed(10)
np.random.seed(10)
torch.manual_seed(10)

######################################################################
######################################################################

CITYSCAPES_MEAN = (0.28689554, 0.32513303, 0.28389177)
CITYSCAPES_STD = (0.18696375, 0.19017339, 0.18720214)

IGNORE_CLASS_LABEL = 19

# Class labels to use for training, found here:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L61
CITYSCAPES_CLASSES_TO_LABELS = {
    0: IGNORE_CLASS_LABEL,
    1: IGNORE_CLASS_LABEL,
    2: IGNORE_CLASS_LABEL,
    3: IGNORE_CLASS_LABEL,
    4: IGNORE_CLASS_LABEL,
    5: IGNORE_CLASS_LABEL,
    6: IGNORE_CLASS_LABEL,
    7: 0,
    8: 1,
    9: IGNORE_CLASS_LABEL,
    10: IGNORE_CLASS_LABEL,
    11: 2,
    12: 3,
    13: 4,
    14: IGNORE_CLASS_LABEL,
    15: IGNORE_CLASS_LABEL,
    16: IGNORE_CLASS_LABEL,
    17: 5,
    18: IGNORE_CLASS_LABEL,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: IGNORE_CLASS_LABEL,
    30: IGNORE_CLASS_LABEL,
    31: 16,
    32: 17,
    33: 18,
    -1: IGNORE_CLASS_LABEL
}

#####################################################################
################################################################

def meanIoU(y_true_all, y_pred_all, num_classes=19):
    num_examples = y_true_all.shape[0]
    mIoU = 0.0
    
    for j in range(num_examples):
        
        y_true = np.squeeze(y_true_all[j,...]).reshape(-1).astype('uint8')
        y_pred = np.squeeze(y_pred_all[j,...]).reshape(-1).astype('uint8')
        
        mIoU_class = np.ones((num_classes,))
        
        clas = np.unique(np.concatenate((np.unique(y_pred), np.unique(y_true)))) 

        for i in range(num_classes):

            if i in clas:

                y_t = np.zeros_like(y_true)
                y_t[y_true == i] = 1
                y_t = y_t.astype('float32')

                y_p = np.zeros_like(y_pred)
                y_p[y_pred == i] = 1
                y_p = y_p.astype('float32')
            
                weight = np.zeros_like(y_true)
                weight[y_true == i] = 1
                weight[y_pred == i] = 1

                intersection = np.sum(y_t * y_p)
                union = np.sum(weight)

                mIoU_class[i] = ((intersection)/(union))

            # print("class: {} mIoU: {}".format(i, ((intersection + 1.0)/(union + 1.0))))
        
        mIoU += (np.sum(mIoU_class) / 20)
        print(clas)
        print("Example: {} mIoU: {}".format(j, np.sum(mIoU_class) / 20))

    return mIoU / num_examples

######################################################################
#############################################################################################


def _get_config():
    
    parser = argparse.ArgumentParser(description="Main handler for training", usage="python ./train.py -g 0")
    parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)
    
    args = parser.parse_args()
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config=0
    return config

###################################################################################


def main(config):

    # device
    device = torch.device("cuda")
    initial_epoch =98 
    num_epochs = 100

    outdir = '/usr/local/data/raghav/ECSE626_2019/Project/Experiments/'              # Full Path to Directory where to store all generated files: Ex. "/usr/local/data/raghav/MSLAQ_experiments/Experiments"
    main_path = '/usr/local/data/raghav/ECSE626_2019/Project/data/'          # Full Path of Input HDf5 file: Ex. "/usr/local/data/raghav/MSLAQ_loader/MSLAQ.hdf5"
    ConfigName = 'Seg'                                                     # Configuration Name to Uniquely Identify this Experiment

    #########################################################################################################

    log_path = join(outdir,ConfigName,'log')

    os.makedirs(log_path,exist_ok=True)
    os.makedirs(join(log_path, 'weights'), exist_ok=True)
    os.makedirs(join(log_path, 'visualize'), exist_ok=True)

    ##################################################################################################

    #####################################################################################################

    model = DeepLab(backbone='resnet', output_stride=8, num_classes=[20], sync_bn=False, freeze_bn=False)

    print("===> Model Defined.")

    #############################################################################################

    params = ([p for p in model.parameters()])

    #############################################################################################


    optimizer = optim.SGD(params, lr=0.0025, momentum=0.9, nesterov=True)
    lambda2 = lambda epoch: (1 - epoch/num_epochs)**0.9
    scheduler = LambdaLR(optimizer, lr_lambda=lambda2)

    print("===> Optimizer Initialized")

    ########################################################################################

    if initial_epoch > 0:
        print("===> Loading pre-trained weight {}".format(initial_epoch))
        weight_path = 'weights/model-{:04d}.pt'.format(initial_epoch)
        # model = torch.load(join(log_path, weight_path))
        checkpoint = torch.load(join(log_path, weight_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']

    #print(model)
    model = model.to(device)

    ##################################################################################################

    input_transform = transforms.Compose([ transforms.Resize(size=128, interpolation=2), PIL_To_Tensor(), NormalizeRange(), transforms.Normalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD) ])
    label_transform = transforms.Compose([ transforms.Resize(size=128, interpolation=0), PIL_To_Tensor(), Project(projection=CITYSCAPES_CLASSES_TO_LABELS)  ])
    depth_transform = transforms.Compose([ transforms.Resize(size=128, interpolation=2), PIL_To_Tensor(), DepthConversion() ])

    training_data_loader = torch.utils.data.DataLoader( Cityscapes(root='/usr/local/data/raghav/ECSE626_2019/Project/data/', split='train', mode='fine', target_type=['semantic'], transform=input_transform, target_transform=[label_transform] ), 
                                                        batch_size=8, 
                                                        shuffle=True, 
                                                        num_workers=4, 
                                                        drop_last=True)

    validation_data_loader = torch.utils.data.DataLoader( Cityscapes(root='/usr/local/data/raghav/ECSE626_2019/Project/data/', split='val', mode='fine', target_type=['semantic'], transform=input_transform, target_transform=[label_transform] ), 
                                                        batch_size=8, 
                                                        shuffle=True, 
                                                        num_workers=4, 
                                                        drop_last=True)

    print("===> Training and Validation Data Loaderss Initialized")

    ##########################################################################################################

    my_metric = ['meanIoU']#, "mse", "mae"]

    my_loss = ["loss"]#, "loss_seg"]

    logger = Logger(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)
    LP = LossPlotter(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)

    print("===> Logger and LossPlotter Initialized")

    ############################################################################################

    def checkpoint(epoch):
        w_path = 'weights/model-{:04d}.pt'.format(epoch)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, join(log_path, w_path))
        print("===> Checkpoint saved to {}".format(w_path))



    #####################################################################################################

    def validate():

        model.eval()

        metric = np.zeros(len(my_metric)+len(my_loss))

        seg_crit = nn.CrossEntropyLoss(reduction='none')
        reg_crit = nn.L1Loss(reduction='none')

        with torch.no_grad():

            for iteration, batch in enumerate(validation_data_loader):

                inp, seg = batch[0].to(device),  batch[1].type('torch.LongTensor').squeeze().to(device) #batch[1][1].to(device), target,

                outp = model(inp)

                loss = 0

                loss_seg = seg_crit(outp[0], seg)
                loss_seg = torch.mean(loss_seg)

                # loss_reg =  reg_crit(outp[1], target)
                # loss_reg = torch.mean(loss_reg)

                # loss = loss_seg + loss_reg
                loss = loss_seg

                loss = loss.item()

                seg = np.squeeze(seg.data.cpu().numpy().astype('float32'))
                outp_seg = torch.argmax(outp[0], dim=1, keepdim=False)
                outp_seg = np.squeeze(outp_seg.data.cpu().numpy().astype('float32'))

                mIoU = meanIoU(seg, outp_seg)

                # mean_squared_error = torch.mean((outp[1]-target)**2).item()

                # mean_absolute_error = torch.mean(torch.abs(outp[1]-target)).item()

                metric += np.array([loss, mIoU])#, mean_squared_error, mean_absolute_error])

                # if iteration==10:
                #     break

        return metric/len(validation_data_loader)
        # return metric/10

    #############################################################################################3
    #####################


    val_metric = validate()
    print("===> Validation Epoch {}: Loss - {:.4f}, mean IoU - {:.4f}".format(initial_epoch, val_metric[0], val_metric[1]))
            # print("===> Validation Epoch {}: Loss - {:.4f}, Loss_Seg - {:.4f}, Loss_Reg - {:.4f}, mean IoU - {:.4f}, MSE - {:.4f}, MAE - {:.4f}".format(epch, val_metric[0], val_metric[1], val_metric[2], val_metric[3], val_metric[4], val_metric[5]))

###################################################################################

if __name__ == "__main__":
    main(_get_config())

