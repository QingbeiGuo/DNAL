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
from models.vgg.vgg_AutoML import VGG_AutoML, vgg16_bn_AutoML
from models.vgg.vgg import vgg16_bn
#dataset
import dataset.imagenet.dataset_imagenet

##############################################################################################################
class FineTuner_CNN:
	def __init__(self, train_path, test_path, model):
	    self.args = args

	    self.epochs = self.args.epochs
	    self.learningrate = self.args.learning_rate
	    self.learning_rate_decay = self.args.learning_rate_decay
	    self.momentum = self.args.momentum
	    self.weight_decay = self.args.weight_decay

	    self.stages = self.args.stages
	    self.epochs_arch = self.args.epochs_arch
	    self.learningrate_arch = self.args.learning_rate_arch
	    self.learning_rate_decay_arch = self.args.learning_rate_decay_arch
	    self.momentum_arch = self.args.momentum_arch
	    self.weight_decay_arch = self.args.weight_decay_arch
	    self.threshold_arch = self.args.threshold_arch
	    self.delta_arch_D = self.args.delta_arch_D

	    self.epochs_param = self.args.epochs_param
	    self.learningrate_param = self.args.learning_rate_param
	    self.learning_rate_decay_param = self.args.learning_rate_decay_param
	    self.momentum_param = self.args.momentum_param
	    self.weight_decay_param = self.args.weight_decay_param

	    self.train_path = self.args.train_path
	    self.test_path = self.args.test_path

	    #imagenet
	    self.train_data_loader = dataset.imagenet.dataset_imagenet.train_loader(self.train_path)
	    self.test_data_loader  = dataset.imagenet.dataset_imagenet.test_loader(self.test_path)

	    self.model = model
	    self.criterion = torch.nn.CrossEntropyLoss()

	    self.accuracys1 = []
	    self.accuracys5 = []

	    self.criterion.cuda()
	    self.model.cuda()

	    for param in self.model.parameters():
	        param.requires_grad = True

	    self.model.train()

##############################################################################################################
	def autoML(self):
		for i in list(range(self.stages)):
		    # first stage, fix scale layer, training parameters
		    for module in self.model.modules():
		        if (isinstance(module, nn.Conv2d)) or (isinstance(module, nn.Linear)) or (isinstance(module, nn.BatchNorm2d)) or (isinstance(module, nn.BatchNorm1d)):
		            for param in module.parameters():
		                param.requires_grad = True

		        if module.__str__().startswith('ScaleLayer2d') or module.__str__().startswith('ScaleLayer1d'):
		            module.scale.data = torch.ones(module.scale.size(0)).cuda()
		            module.delta = torch.ones(module.delta.size(0)).cuda() * self.delta_arch_D
		            module.ignore = torch.tensor(2).cuda()
		            for param in module.parameters():
		                param.requires_grad = False

		    self.accuracys1 = torch.load("accuracys1_trainning")
		    self.accuracys5 = torch.load("accuracys5_trainning")
		    self.model = torch.load("model_training_param29").cuda()
		    print("self.accuracys1", self.accuracys1)
		    print("self.accuracys5", self.accuracys5)

		    # training parameters
		    for j in list(range(self.epochs_param[i])):
		        print("Epoch_Param: ", j)

		        optimizer = optim.SGD(
		                              filter(lambda p: p.requires_grad, self.model.parameters()), 
		                              lr=self.learningrate_param, 
		                              momentum=self.momentum_param, 
		                              weight_decay=self.weight_decay_param)
		        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

		        self.train_epoch_param(j, -1, optimizer, scheduler)
		        cor1, cor5 = self.test()

		        torch.save(self.model, "model_training_param" + str(j))
		        torch.save(self.accuracys1, "accuracys1_trainning")
		        torch.save(self.accuracys5, "accuracys5_trainning")

		        self.adjust_learning_rate_param(i,j)

###############################################################

		    #  second stage, fix parametersm, training scale layer
		    for module in self.model.modules():
		        if (isinstance(module, nn.Conv2d)) or (isinstance(module, nn.BatchNorm2d)):
		            for param in module.parameters():
		                param.requires_grad = False

		        if module.__str__().startswith('ScaleLayer2d'):
		            module.scale.data = torch.zeros(module.scale.size(0)).cuda()
		            module.delta = torch.ones(module.delta.size(0)).cuda() * self.delta_arch_D
		            module.ignore = torch.tensor(0).cuda()
		            for param in module.parameters():
		                param.requires_grad = True

		    self.learningrate_arch = self.args.learning_rate_arch
		    self.delta_arch_D = self.args.delta_arch_D
		    for k in list(range(self.epochs_arch[i])):
		        print("Epoch_Arch: ", k)
		        self.adjust_learning_rate_arch(i,k)
		        self.adjust_delta_arch(i,k)

		        optimizer = optim.SGD(
		                              filter(lambda p: p.requires_grad, self.model.parameters()), 
		                              lr=self.learningrate_arch, 
		                              momentum=self.momentum_arch,
		                              weight_decay=self.weight_decay_arch)
		        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

		        self.train_epoch_arch(k, -1, optimizer, scheduler)
		        cor1, cor5 = self.test()

		        torch.save(self.model, "model_training_arch" + str(k))
		        torch.save(self.accuracys1, "accuracys1_trainning")
		        torch.save(self.accuracys5, "accuracys5_trainning")

		        scale_arch = []
		        for module in self.model.modules():
		            if module.__str__().startswith('ScaleLayer2d'):
		                #print("torch.sigmoid(module.scale)", torch.sigmoid(module.delta*module.scale))
		                scale_arch = scale_arch + torch.sigmoid(module.delta*module.scale).data.cpu().tolist()
		        print("scale_arch1", scale_arch)

		    scale_arch = []
		    for module in self.model.modules():
		        if module.__str__().startswith('ScaleLayer2d'):
		            #print("torch.sigmoid(module.scale)", torch.sigmoid(module.delta*module.scale))
		            scale_arch = scale_arch + torch.sigmoid(module.delta*module.scale).data.cpu().tolist()
		    print("scale_arch1", scale_arch)

################################################################

		# prune model
		for index, module in enumerate(self.model.modules()):
		    if module.__str__().startswith('ScaleLayer2d'):
		        layer_mask = list(np.where(np.array(torch.sigmoid(module.delta*module.scale).data.cpu().tolist()) > self.threshold_arch, 1, 0))
		        layer_combine_mask = [layer_mask[i] and module._mask.tolist()[i] for i in range(min(len(layer_mask),len(module._mask)))]
		        module.mask = torch.Tensor(layer_combine_mask).cuda()
		        #print("mask", module.mask)

		        if module.__str__().startswith('ScaleLayer2d'):
		            self.model.set_conv_mask(index-2, np.where(np.array(layer_combine_mask) == 0))

		cor1, cor5 = self.test()

		torch.save(self.model, "model_training")
		torch.save(self.accuracys1, "accuracys1_trainning")
		torch.save(self.accuracys5, "accuracys5_trainning")

###########################################################

		# third stage, fix scale layer, training parameters
		for module in self.model.modules():
		    if (isinstance(module, nn.Conv2d)) or (isinstance(module, nn.Linear)) or (isinstance(module, nn.BatchNorm2d)) or (isinstance(module, nn.BatchNorm1d)):
		        for param in module.parameters():
		            param.requires_grad = True

		    if module.__str__().startswith('ScaleLayer2d'):
		        module.ignore = torch.tensor(1).cuda()
		        for param in module.parameters():
		            param.requires_grad = False

		accuracy = 0
		for i in list(range(self.epochs)):
		    print("Epoch: ", i)

		    optimizer = optim.SGD(
		                          filter(lambda p: p.requires_grad, self.model.parameters()), 
		                          lr=self.learningrate, 
		                          momentum=self.momentum,
		                          weight_decay=self.weight_decay)
		    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

		    self.train_epoch(i, -1, optimizer, scheduler)
		    cor1, cor5 = self.test()

		    #save the best model
		    if cor1 > accuracy:
		        torch.save(self.model, "model_training_m")
		        accuracy = cor1

		    torch.save(i, "epoch_i")
		    torch.save(self.model, "model_training_" + str(i))
		    #torch.save(self.model, "model_training_")
		    torch.save(self.accuracys1, "accuracys1_trainning")
		    torch.save(self.accuracys5, "accuracys5_trainning")

		    self.adjust_learning_rate(i)

###########################################################

	def train_epoch_param(self, epoch, batches, optimizer = None, scheduler = None):
		for step, (batch, label) in enumerate(self.train_data_loader):
		    if (step == batches):
		        break
		    self.train_batch_param(epoch, step, batch, label, optimizer, scheduler)

	def train_batch_param(self, epoch, step, batch, label, optimizer = None, scheduler = None):
		### Compute output
		batch,label = Variable(batch.cuda()),Variable(label.cuda())                   #Tensor->Variable
		output = self.model(batch)
		loss = self.criterion(output, label)

		if step % self.args.print_freq == 0:
		    print("Epoch-step: ", epoch, "-", step, ":", loss.data.cpu().numpy())

		### Compute gradient and do SGD step
		self.model.zero_grad()
		loss.backward()
		optimizer.step()                                                              #update parameters

	def train_epoch_arch(self, epoch, batches, optimizer = None, scheduler = None):
		for step, (batch, label) in enumerate(self.train_data_loader):
		    if (step == batches):
		        break
		    self.train_batch_arch(epoch, step, batch, label, optimizer, scheduler)

	def train_batch_arch(self, epoch, step, batch, label, optimizer = None, scheduler = None):
		### Compute output
		batch,label = Variable(batch.cuda()),Variable(label.cuda())                   #Tensor->Variable
		output = self.model(batch)
		loss_ = self.criterion(output, label)

		### Add scale layer loss
		arch_loss = 0
		if args.lambda_arch > 0:
		    for module in self.model.modules():
		        if module.__str__().startswith('ScaleLayer'):
		            #print("module.scale:", module.scale)
		            arch_loss = arch_loss + torch.sigmoid(module.scale).sum()

		loss = loss_ + args.lambda_arch * arch_loss
		if step % self.args.print_freq == 0:
		    print("Epoch-step: ", epoch, "-", step, ":", loss.data.cpu().numpy(), loss_.data.cpu().numpy(), args.lambda_arch * arch_loss.data.cpu().numpy())

#		loss = self.criterion(output, label)
#		if step % self.args.print_freq == 0:
#		    print("Epoch-step: ", epoch, "-", step, ":", loss)

		### Compute gradient and do SGD step
		self.model.zero_grad()
		loss.backward()
		optimizer.step()                                                              #update parameters

	def train_epoch(self, epoch, batches, optimizer = None, scheduler = None):
		for step, (batch, label) in enumerate(self.train_data_loader):
		    if (step == batches):
		        break
		    self.train_batch(epoch, step, batch, label, optimizer, scheduler)

	def train_batch(self, epoch, step, batch, label, optimizer = None, scheduler = None):
		### Compute output
		batch,label = Variable(batch.cuda()),Variable(label.cuda())                   #Tensor->Variable
		output = self.model(batch)
		loss = self.criterion(output, label)

		if step % self.args.print_freq == 0:
		    print("Epoch-step: ", epoch, "-", step, ":", loss.data.cpu().numpy())

		### Compute gradient and do SGD step
		self.model.zero_grad()
		loss.backward()
		optimizer.step()                                                              #update parameters

###########################################################

	def test(self, flag = -1):
		self.model.eval()

		correct1 = 0
		correct5 = 0
		total = 0

		print("Testing...")
		for i, (batch, label) in enumerate(self.test_data_loader):
			  batch,label = Variable(batch.cuda()),Variable(label.cuda())              #Tensor->Variable
			  output = self.model(batch)
			  cor1, cor5 = accuracy(output.data, label, topk=(1, 5))                   # measure accuracy top1 and top5
			  correct1 += cor1
			  correct5 += cor5
			  total += label.size(0)

		if flag == -1:
		    self.accuracys1.append(float(correct1) / total)
		    self.accuracys5.append(float(correct5) / total)

		print("Accuracy Top1:", float(correct1) / total)
		print("Accuracy Top5:", float(correct5) / total)

		self.model.train()                                                              

		return float(correct1) / total, float(correct5) / total

##############################################################################################################

	def adjust_learning_rate(self, epoch):
        #manually
		if self.args.learning_rate_decay == 0:
		    #imagenet
		    if epoch in [10, 40]:
		        self.learningrate = self.learningrate/10;

		    self.learningrate = self.learningrate * lr_decay
		print("self.learningrate", self.learningrate)

	def adjust_learning_rate_arch(self, epoch_i, epoch_k):
        #manually
		if self.args.learning_rate_decay == 0:
		    #imagenet
		    if epoch_k in []:
		        self.learningrate_arch = self.learningrate_arch/10;

		    self.learningrate_arch = self.learningrate_arch * lr_decay
		print("self.learningrate_arch", self.learningrate_arch)

	def adjust_learning_rate_param(self, epoch_i, epoch_j):
        #manually
		if self.args.learning_rate_decay == 0:
		    #imagenet
		    self.learningrate_param = 0.01;

		    self.learningrate_param = self.learningrate_param * lr_decay
		print("self.learningrate_param", self.learningrate_param)

	def adjust_delta_arch(self, epoch_i, epoch_k):
		num_epochs = 10
		delta_start = 1
		#print("lr_start = "+str(self.lr_start))
		delta_fin = 10000
		#print("lr_fin = "+str(self.lr_fin))
		delta_decay = (delta_fin/delta_start)**(1./num_epochs)
		#print("lr_decay = "+str(self.lr_decay))

		self.delta_arch_D = self.delta_arch_D * delta_decay
		print("self.delta_arch_D", self.delta_arch_D)

		for module in self.model.modules():
		    if module.__str__().startswith('ScaleLayer2d') or module.__str__().startswith('ScaleLayer1d'):
		        module.delta = torch.ones(module.delta.size(0)).cuda() * self.delta_arch_D

##############################################################################################################

def accuracy(output, target, topk=(1,)):                                               
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)                                          
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))                               

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)                                 
        res.append(correct_k)
    return res

##############################################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CNN Training')

    parser.add_argument('--arch', '--a', default='VGG16', help='model architecture: (default: VGG16)')

    parser.add_argument('--epochs', type=int, default=70, help='number of total epochs to run')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.01, help = 'initial learning rate')
    parser.add_argument('--learning_rate_decay', '--lr_decay', type=int, default=0, help = 'maually[0] or exponentially[1] decaying learning rate')
    parser.add_argument('--momentum', '--mm', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

    parser.add_argument('--stages', type=int, default=1, help='number of total epochs to run')
    parser.add_argument('--epochs_arch', type=int, default=[10], help='number of total epochs to search architecture')
    parser.add_argument('--learning_rate_arch', '--lr_arch', type=float, default=0.01, help = 'initial learning rate')
    parser.add_argument('--learning_rate_decay_arch', '--lr_decay_arch', type=int, default=0, help = 'maually[0] or exponentially[1] decaying learning rate')
    parser.add_argument('--momentum_arch', '--mm_arch', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay_arch', '--wd_arch', type=float, default=0, help='weight decay (default: 1e-4)')
    parser.add_argument('--lambda_arch', '--lambda_arch', type=float, default=1e-4, help = 'lambda')
    parser.add_argument('--threshold_arch', '--threshold_arch', type=float, default=0.9, help='threshold (default: 0.1)')
    parser.add_argument('--delta_arch_D', '--delta_arch_D', type=float, default=1, help='delta (default: 1)')

    parser.add_argument('--epochs_param', type=int, default=[30], help='number of total epochs to optimize parameter')
    parser.add_argument('--learning_rate_param', '--lr_param', type=float, default=0.01, help = 'initial learning rate')
    parser.add_argument('--learning_rate_decay_param', '--lr_decay_param', type=int, default=0, help = 'maually[0] or exponentially[1] decaying learning rate')
    parser.add_argument('--momentum_param', '--mm_param', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay_param', '--wd_param', type=float, default=1e-4, help='weight decay (default: 1e-4)')

    parser.add_argument('--print_freq', '--p', type=int, default=100, help = 'print frequency (default:20)')
    #imagenet
    parser.add_argument('--train_path',type=str, default='/data1/Datasets/ImageNet/ILSVRC2012/ILSVRC2012_img_train/', help = 'train dataset path')
    parser.add_argument('--test_path', type=str, default='/data1/Datasets/ImageNet/ILSVRC2012/ILSVRC2012_img_val_subfolders/', help = 'test dataset path')
    parser.add_argument("--parallel", type = int, default = 1)
    parser.set_defaults(autoML=True)
    parser.set_defaults(train=False)
    args = parser.parse_args()

    return args

##############################################################################################################
if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

	args = get_args()
	print("args:", args)

	model = vgg16_bn_AutoML().cuda()
	torch.save(model, "model")
	print("model_training:", model)

	if args.parallel == 1:
	    model = torch.nn.DataParallel(model).cuda()

	fine_tuner = FineTuner_CNN(args.train_path, args.test_path, model)
	fine_tuner.test()

	if args.autoML:
	    fine_tuner.autoML()