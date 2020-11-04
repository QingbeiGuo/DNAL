#!/usr/bin/python
# -*- coding: UTF-8 -*-

#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import os
import math
#model
from models.mobilenet.mobilenetv2_AutoML import MobileNetV2_AutoML, mobilenetv2_AutoML
#dataset
import dataset.cifar10.dataset_cifar10

##############################################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CNN Training')
    parser.add_argument('--arch', '--a', default='MobileNet', help='model architecture: (default: MobileNet)')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    return args

##############################################################################################################
if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	args = get_args()
	print("args:", args)

	model = mobilenetv2_AutoML().cuda()
	model = torch.load("model_training").module
	print("model_training:", model)

	i=0
	for index, module in enumerate(model.modules()):
	    if module.__str__().startswith('ScaleLayer2d'):
	        #print("module", module)
	        mask = module._mask.cpu().tolist()  #[out_channels, in_channels, W, H]
	        #print("mask conv",mask)
	        #print("index, channel_num, mask_num, channel_num-mask_num, percent {:.2f}%", index,len(mask),np.sum(mask==0),len(mask)-np.sum(mask==0),format(np.sum(mask==0)/len(mask)*100))
	        print(mask.count(1)+mask.count(0),mask.count(1),mask.count(0))
	        i+=1

	print("i", i)

	# prune model
	for index, module in enumerate(model.modules()):
	    print("index, module",index, module)
	    if module.__str__().startswith('ScaleLayer2d'):
	        layer_mask = list(np.where(np.array(torch.sigmoid(module.delta*module.scale).data.cpu().tolist()) > 0.9, 1, 0))
	        layer_combine_mask = [layer_mask[i] and module._mask.tolist()[i] for i in range(min(len(layer_mask),len(module._mask)))]
	        module.mask = torch.Tensor(layer_combine_mask).cuda()
	        #print("mask", module.mask)
	        print("index, module",index, module)

	        if module.__str__().startswith('ScaleLayer2d'):
	            print("module",module)
	            print("index-2, np.where(np.array(layer_combine_mask) == 0)[0]",index-2, np.where(np.array(layer_combine_mask) == 0)[0])
	            model.set_conv_mask(index-2, np.where(np.array(layer_combine_mask) == 0)[0])

	torch.save(model, "model_training_")