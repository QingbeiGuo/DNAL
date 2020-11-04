# Weak Sub-network Pruning for Strong Sub-networks of Deep Neural Networks

## Introduction

Although pruning methods have recently attracted ever-increasing focus to compress and accelerate deep convolutional neural networks (CNNs), which significantly allows the pruned networks to be deployed on the resource-constrained hardware devices, there are still three intractable problems: (1) neglecting the correlation of the pruned channels between layers; (2) ignoring the performance restorability of the neural network models after pruning; (3) mostly requiring time-consuming iterative pruning and fine-tuning processes. To address these issues, we theoretically prove the relationship between the activation and gradient sparsity and the channel saliency, and based on the theoretical results, propose a novel and effective method of \emph{weak sub-network pruning} (WSP). Specifically, for a well-trained network model, we divide the whole compression process into two non-iterative stages. The first stage is to directly obtain a strong sub-network by pruning the weakest one. We first select the unimportant channels from all the layers to pattern the weakest sub-network, where each selected channel makes a minimal contribution in both the feed-forward and feed-back processes. Then, a one-shot pruning strategy is conducted for the weakest sub-network to achieve a strong one with good accuracy restorability, while significantly reducing the impact of network depth and width on the compression efficiency, especially for deep and wide network architectures. The other stage is to globally fine-tune the strong sub-network with a certain number of epoches to restore its original recognition accuracy. Furthermore, our proposed method adapts to the fully-connected layers as well as the convolutional layers for simultaneous compression and acceleration. The comprehensive experiments on VGG16 and ResNet50 over a variety of popular benchmarks, such as ImageNet-1K, CUB-200 and PASCAL VOC, demonstrate that our WSP method achieves superior performance on classification, domain adaption and object detection with small model size, low computation complexity and low energy consumption. Our source code is available at \url{https://github.com/QingbeiGuo/WSP.git}.

This project is a pytorch implementation, aiming to compressing and accelerating deep convolutional neural networks. 

### What we are doing and going to do

- [x] Support pytorch-1.0.

## Classification

We benchmark our code thoroughly on imagenet-1K for classification, using two different network architecture: VGG16 and Resnet50. Below are the results:

1) VGG16 on ImageNet:

Comparison among several state-of-the-art methods for VGG16 on ILSVRC2012

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
Baseline               | 138.37M | 15.47G   | 71.59   | 90.38
WSP (Conv-20/FC-20)    | 112.06M | 11.74G   | 72.24   | 90.90
WSP (Conv-30/FC-30)    |  96.62M |  9.56G   | 71.75   | 90.64
WSP (Conv-40/FC-40)    |  82.06M |  7.61G   | 71.08   | 90.17
WSP (Conv-50/FC-50)    |  67.70M |  5.76G   | 69.88   | 89.57

2) ResNet on ImageNet:

Comparison among several state-of-the-art methods for ResNet50 on ILSVRC2012

model    | Params  | FLOPs | Top-1 (%) | Top-5 (%)
---------|---------|-------|-----------|-----------
Baseline           | 25.55M|  4.09G| 76.13| 92.86
WSP (Conv-40/FC-40)| 17.12M|  2.51G| 75.49| 92.57
WSP (Conv-50/FC-50)| 14.29M|  2.02G| 74.95| 92.34
WSP (Conv-60/FC-60)| 11.60M|  1.55G| 73.91| 91.66
WSP (Conv-70/FC-70)|  9.07M|  1.12G| 72.04| 90.82

## Domain Adaptation

Comparison of different compressed models for fine-grained classification on CUB-200

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
Baseline                     | 135.09M|  15.47G| 77.68| 94.96
VGG-WSP (Conv-20/FC-20)      | 109.38M|  11.73G| 78.13| 95.06
VGG-WSP (Conv-30/FC-30)      |  94.21M|   9.56G| 77.82| 94.86
VGG-WSP (Conv-40/FC-40)      |  79.97M|   7.60G| 77.60| 94.63
VGG-WSP (Conv-50/FC-50)      |  65.95M|   5.76G| 76.73| 94.43
---------|--------|-------|-----------|-----------
Baseline                     |  23.91M|   4.09G| 79.10| 95.81
ResNet-WSP (Conv-40/FC-40)   |  15.53M|   2.51G| 78.22| 95.74
ResNet-WSP (Conv-50/FC-50)   |  12.72M|   2.02G| 78.05| 95.34
ResNet-WSP (Conv-60/FC-60)   |  10.06M|   1.55G| 76.67| 94.94
ResNet-WSP (Conv-70/FC-70)   |   7.56M|   1.12G| 75.56| 94.41
---------|--------|-------|-----------|-----------
AlexNet [1] (Our impl.)      |  57.82M| 710.92M| 56.80| 82.40
GoogLeNet [53] (Our impl.)   |   5.79M|   1.50G| 61.41| 88.82
MobileNet v2 [54] (Our impl.)|   2.45M| 299.75M| 75.01| 93.87

## Object Detection

Object detection results on MS COCO. Here, mAP-1 and mAP-2 correspond to 300$\times$ and 600$\times$ input resolutions, respectively. mAP is reported with COCO primary challenge metric (AP@IoU=0.50:0.05:0.95)

model    | Params | FLOPs | mAP (%) 
---------|--------|-------|-----------
Baseline                  |  138.37M| 15.47G| 66.82
VGG-WSP (Conv-20/FC-20)   |  112.06M| 11.74G| 65.88
VGG-WSP (Conv-30/FC-30)   |   96.62M|  9.56G| 65.22
VGG-WSP (Conv-40/FC-40)   |   82.06M|  7.61G| 64.75
VGG-WSP (Conv-50/FC-50)   |   67.70M|  5.76G| 63.04
---------|--------|-------|-----------
Baseline                  |   23.50M|  4.09G| 70.30
ResNet-WSP (Conv-40/FC-40)|   17.12M|  2.51G| 73.22
ResNet-WSP (Conv-50/FC-50)|   12.72M|  2.02G| 72.13
ResNet-WSP (Conv-60/FC-60)|   10.06M|  1.55G| 70.82
ResNet-WSP (Conv-70/FC-70)|    7.56M|  1.12G| 69.50

## Train for classification

(1) set compressed = true and fine-tuning = false, pruning the pre-trained models, getting the pruned models; 

(2) set compressed = false and fine-tuning = true, load the pruned models, globally fine-tuning the pruned models.

## Authorship

This project is contributed by [Qingbei Guo](https://github.com/QingbeiGuo).

## Citation
