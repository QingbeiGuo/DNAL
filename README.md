# Differentiable Neural Architecture Learning for Efficient Neural Network Design

## Introduction

Automated neural network design has received ever-increasing attention with the evolution of deep convolutional neural networks (CNNs), especially involving their deployment on embedded and mobile platforms.
One of the biggest problems that neural architecture search (NAS) confronts is that a large number of candidate neural architectures are required to train,  using, for instance, reinforcement learning and evolutionary optimisation algorithms, at a vast computation cost. Even recent differentiable neural architecture search (DNAS) samples a small number of candidate neural architectures based on the probability distribution of learned architecture parameters to select the final neural architecture.
To address this computational complexity issue, we introduce a novel \emph{architecture parameterisation} based on \emph{scaled sigmoid function}, and propose a general \emph{Differentiable Neural Architecture Learning} (DNAL) method to optimize the neural architecture without the need to evaluate candidate neural networks.
Specifically, for stochastic supernets as well as conventional CNNs, we build a new channel-wise module layer with the architecture components controlled by a scaled sigmoid function. We train these neural network models from scratch. The network optimization is decoupled into the weight optimization and the architecture optimization, which avoids the interaction between the two types of parameters and alleviates the vanishing gradient problem. We address the non-convex optimization problem of neural architecture by the continuous scaled sigmoid method with convergence guarantees.
Extensive experiments demonstrate our DNAL method delivers superior performance in terms of neural architecture search cost, and adapts to conventional CNNs (e.g., VGG16 and ResNet50), lightweight CNNs (e.g., MobileNetV2) and stochastic supernets (e.g., ProxylessNAS). The optimal networks learned by DNAL surpass those produced by the state-of-the-art methods on the benchmark CIFAR-10 and ImageNet-1K dataset in accuracy, model size and computational complexity. Our source code is available at \url{https://github.com/QingbeiGuo/DNAL.git}.

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
