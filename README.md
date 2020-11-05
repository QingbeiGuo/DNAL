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

We benchmark our code thoroughly on CIFAR-10 and imagenet-1K for classification, using conventional CNNs (e.g., VGG16 and ResNet50), lightweight CNNs (e.g., MobileNetV2) and stochastic supernets (e.g., ProxylessNAS). Below are the results:

1. CIFAR-10

1) VGG16 on CIFAR-10:

Comparison among several state-of-the-art methods for VGG16 on CIFAR-10

model    | FLOPs(M) | Params(M) | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline]                      | 313.47(1.00$\times$)   | 14.99(1.00$\times$)  | 93.77           | 99.73  
[DNAL($\lambda_a$=1e-5)]        | 211.89(1.48$\times$)   |  5.51(2.72$\times$)  | 93.82           | 99.71  
[DNAL($\lambda_a$=5e-5)]        | 195.14(1.61$\times$)   |  3.73(4.02$\times$)  | 93.53           | 99.77  
[Variational-pruning]           |    190(1.65$\times$)   |  3.92(3.82$\times$)  | 93.18           | -      
[GAL-0.1]                       | 171.89(1.82$\times$)   |  2.67(5.61$\times$)  | 93.42           | -      
[DNAL($\lambda_a$=1e-4)]        | 161.97(1.94$\times$)   |  2.10(7.14$\times$)  | 93.75           | 99.72  
[HRank]                         |  73.70(4.25$\times$)   |  1.78(8.42$\times$)  | 91.23           | -      
[DNAL($\lambda_a$=2e-4)]        |  61.23(5.12$\times$)   |  0.60(24.98$\times$) | 92.33           | 99.69 
[DNAL($\lambda_a$=3e-4)]        |  29.77(10.53$\times$)  |  0.29(51.69$\times$) | 89.93           | 99.62  
[DNAL($\lambda_a$=4e-4)]        |  22.04(14.22$\times$)  |  0.24(62.46$\times$) | 89.92           | 99.41  
[DNAL($\lambda_a$=5e-4)]        |  16.65(18.83$\times$)  |  0.17(88.18$\times$) | 89.27           | 99.51  

2) ResNet56 on CIFAR-10:

Comparison among several state-of-the-art methods for ResNet56 on CIFAR-10

model    | FLOPs(M) | Params(M) | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline]                      | 125.49(1.00$\times$)  |  0.85(1.00$\times$)   | 94.15           | 99.91  
[DNAL($\lambda_a$=1e-5)]        |  93.94(1.34$\times$)  |  0.66(1.29$\times$)   | 93.76           | 99.91  
[HRank]                         |  88.72(1.41$\times$)  |  0.71(1.20$\times$)   | 93.52           | -      
[DNAL($\lambda_a$=5e-5)]        |  83.11(1.51$\times$)  |  0.59(1.44$\times$)   | 93.75           | 99.87  
[NISP]                          |  81.00(1.55$\times$)  |  0.49(1.73$\times$)   | 93.01           | -      
[AMC]                           |   62.7(2.00$\times$)  |                  -    |  91.9           | -      
[KSE(G=4)]                      |     60(2.09$\times$)  |  0.43(1.98$\times$)   | 93.23           | -      
[KSE(G=5)]                      |     50(2.51$\times$)  |  0.36(2.36$\times$)   | 92.88           | -      
[GAL-0.8]                       |  49.99(2.51$\times$)  |  0.29(2.93$\times$)   | 91.58           | -      
[DNAL($\lambda_a$=1e-4)]        |  36.94(3.40$\times$)  |  0.25(3.40$\times$)   | 93.20           | 99.89  
[HRank]                         |  32.52(3.86$\times$)  |  0.27(3.15$\times$)   | 90.72           | -      
[DNAL($\lambda_a$=2e-4)]        |   8.63(14.54$\times$) | 0.060(14.17$\times$)  | 89.31           | 99.66  
[DNAL($\lambda_a$=3e-4)]        |   3.44(36.48$\times$) | 0.022(38.64$\times$)  | 85.83           | 99.45  
[DNAL($\lambda_a$=4e-4)]        |   2.38(52.73$\times$) | 0.013(65.38$\times$)  | 84.07           | 99.31  
[DNAL($\lambda_a$=5e-4)]        |   1.68(74.70$\times$) |0.007(121.43$\times$)  | 83.48           | 99.19  

3) MobileNetV2 on CIFAR-10:

Comparison among several state-of-the-art methods for MobileNetV2 on CIFAR-10

model    | FLOPs(M) | Params(M) | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline]                      |  91.17(1.00$\times$) |  2.30(1.00$\times$)  | 94.31           | 99.90  
[FLGC(G=2)]                     |     79(1.15$\times$) |  1.18(1.95$\times$)  | 94.11           | -      
[FLGC(G=3)]                     |     61(1.49$\times$) |  0.85(2.71$\times$)  | 94.20           | -      
[DNAL($\lambda_a$=1e-5)]        |  59.47(1.53$\times$) |  1.43(1.61$\times$)  | 94.17           | 99.89  
[DNAL($\lambda_a$=5e-5)]        |  54.98(1.66$\times$) |  1.20(1.92$\times$)  | 94.30           | 99.86  
[FLGC(G=4)]                     |   51.5(1.77$\times$) |  0.68(3.38$\times$)  | 94.16           | -      
[FLGC(G=5)]                     |     46(1.98$\times$) |  0.58(3.97$\times$)  | 93.88           | -      
[FLGC(G=6)]                     |   42.5(2.15$\times$) |  0.51(4.51$\times$)  | 93.67           | -      
[FLGC(G=7)]                     |     40(2.28$\times$) |  0.46(5.00$\times$)  | 93.66           | -      
[FLGC(G=8)]                     |     38(2.40$\times$) |  0.43(5.35$\times$)  | 93.09           | -      
[DNAL($\lambda_a$=1e-4)]        |  36.63(2.49$\times$) |  0.65(3.54$\times$)  | 94.01           | 99.89  
[DNAL($\lambda_a$=2e-4)]        |  13.35(6.83$\times$) | 0.20(11.50$\times$)  | 91.96           | 99.91  
[DNAL($\lambda_a$=3e-4)]        |  7.81(11.67$\times$) | 0.12(19.17$\times$)  | 90.65           | 99.82  
[DNAL($\lambda_a$=4e-4)]        |  5.40(16.88$\times$) |0.096(23.96$\times$)  | 88.83           | 99.76  
[DNAL}($\lambda_a$=5e-4)]       |  4.50(20.26$\times$) |0.081(28.40$\times$)  | 87.85           | 99.62  

2. ImageNet-1K

1) VGG16 on ImageNet-1K:

Comparison among several state-of-the-art methods for VGG16 on ImageNet-1K

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline](https://pan.baidu.com/s/10ofp_aLnX5RCisDsVzXT-Q)                    | 25.55M | 4.09G   | 76.13   | 92.862
[ResNet-G (Conv-60/FC-60)](https://pan.baidu.com/s/10ofp_aLnX5RCisDsVzXT-Q)    | 11.88M | 1.91G   | 75.20   | 92.55
[ResNet-G (Conv-70/FC-60)](https://pan.baidu.com/s/10ofp_aLnX5RCisDsVzXT-Q)    |  9.83M | 1.55G   | 74.43   | 92.30
[ResNet-G (Conv-80/FC-60)](https://pan.baidu.com/s/10ofp_aLnX5RCisDsVzXT-Q)    |  7.76M | 1.20G   | 73.22   | 91.70
[ResNet-LG (Conv-60/FC-60)](https://pan.baidu.com/s/10ofp_aLnX5RCisDsVzXT-Q)   | 11.87M | 1.91G   | 75.12   | 92.59
[ResNet-LG (Conv-70/FC-60)](https://pan.baidu.com/s/10ofp_aLnX5RCisDsVzXT-Q)   |  9.83M | 1.56G   | 74.42   | 92.31
[ResNet-LG (Conv-80/FC-60)](https://pan.baidu.com/s/10ofp_aLnX5RCisDsVzXT-Q)   |  7.76M | 1.20G   | 73.38   | 91.69

    \specialrule{0.10em}{0pt}{0pt}
    Baseline                            &  15.47(1.00$\times$)  & 138.37(1.00$\times$)  & 76.13  & 92.86  & 90    
    \specialrule{0.08em}{0pt}{0pt}
    GDP~\cite{LinJLWHZ18}               &    7.5(2.06$\times$)  &                    -  & 69.88  & 89.16  & 90+30+20    
    GDP~\cite{LinJLWHZ18}               &    6.4(2.42$\times$)  &                    -  & 68.80  & 88.77  & 90+30+20    
    ThiNet~\cite{LuoZZXWL18}            &   4.79(3.23$\times$)  & 131.44(1.05$\times$)  & 69.74  & 89.41  & 196+48     
    \textbf{DNAL}($\lambda_a$=1e-4)     &   4.69(3.30$\times$)  &  77.05(1.80$\times$)  & 69.80  & 89.42  & 30+10+70
    SSR-L2,1~\cite{LinJLDL19}           &    4.5(3.44$\times$)  &  126.7(1.09$\times$)  & 69.80  & 89.53  & -     
    SSR-L2,0~\cite{LinJLDL19}           &    4.5(3.44$\times$)  &  126.2(1.10$\times$)  & 69.99  & 89.42  & -     
    GDP~\cite{LinJLWHZ18}               &    3.8(4.07$\times$)  &                    -  & 67.51  & 87.95  & 90+30+20     
    SSS~\cite{HuangW18}                 &    3.8(4.07$\times$)  &  130.5(1.06$\times$)  & 68.53  & 88.20  & 100     
    ThiNet~\cite{LuoZZXWL18}            &   3.46(4.47$\times$)  & 130.50(1.06$\times$)  & 69.11  & 88.86  & 196+48     
    \specialrule{0.10em}{0pt}{0pt}

2) ResNet50 on ImageNet-1K:

Comparison among several state-of-the-art methods for ResNet50 on ImageNet-1K

    \specialrule{0.10em}{0pt}{0pt}
    Baseline                           &  4.09(1.00$\times$)  & 25.55(1.00$\times$)  & 75.19  & 92.56  & 90    \\
    \specialrule{0.08em}{0.5pt}{0pt}
    \textbf{DNAL}($\lambda_a$=5e-5)    &  2.07(1.98$\times$)  & 15.34(1.67$\times$)  & 74.07  & 92.02  & 30+10+70
    SSR-L2,1~\cite{LinJLDL19}          &   1.9(2.15$\times$)  &  15.9(1.61$\times$)  & 72.13  & 90.57  & -  
    SSR-L2,0~\cite{LinJLDL19}          &   1.9(2.15$\times$)  &  15.5(1.65$\times$)  & 72.29  & 90.73  & -  
    GDP~\cite{LinJLWHZ18}              &  1.88(2.18$\times$)  &                   -  & 71.89  & 90.71  & 90+30+20   
    GAL-0.5-joint~\cite{LinJYZCYHD19}  &  1.84(2.22$\times$)  & 19.31(1.32$\times$)  & 71.80  & 90.82  & 90+60     
    ABCPruner~\cite{LinJZZWT20}        &  1.79(2.28$\times$)  & 11.24(2.27$\times$)  & 73.52  & 91.51  & 12+90
    \textbf{DNAL}($\lambda_a$=6e-5)    &  1.75(2.34$\times$)  & 12.75(2.00$\times$)  & 73.65  & 91.74  & 30+10+70
    ThiNet-50~\cite{LuoZZXWL18}        &  1.71(2.39$\times$)  & 12.38(2.06$\times$)  & 72.03  & 90.99  & 196+48
    SSR-L2,1~\cite{LinJLDL19}          &   1.7(2.41$\times$)  &  12.2(2.09$\times$)  & 71.15  & 90.29  & -     
    SSR-L2,0~\cite{LinJLDL19}          &   1.7(2.41$\times$)  &  12.0(2.13$\times$)  & 71.47  & 90.19  & -     
    GAL-1~\cite{LinJYZCYHD19}          &  1.58(2.59$\times$)  & 14.67(1.74$\times$)  & 69.88  & 89.75  & 90+60
    GDP~\cite{LinJLWHZ18}              &  1.57(2.61$\times$)  &                   -  & 70.93  & 90.14  & 90+30+20
    HRank~\cite{LinJWZZTL20}           &  1.55(2.64$\times$)  & 13.77(1.86$\times$)  & 71.98  & 91.01  & -     
    \textbf{DNAL}($\lambda_a$=7e-5)    &  1.44(2.84$\times$)  & 10.94(2.34$\times$)  & 72.86  & 91.34  & 30+10+70
    ABCPruner~\cite{LinJZZWT20}        &  1.30(3.15$\times$)  &                   -  & 72.58  & -      & 12+90
    GAL-1-joint~\cite{LinJYZCYHD19}    &  1.11(3.68$\times$)  & 10.21(2.50$\times$)  & 69.31  & 89.12  & 90+60
    ThiNet-30~\cite{LuoZZXWL18}        &  1.10(3.72$\times$)  &  8.66(2.95$\times$)  & 68.17  & 88.86  & 196+48
    HRank~\cite{LinJWZZTL20}           &  0.98(4.17$\times$)  &  8.27(3.09$\times$)  & 69.10  & 89.58  & -    
    ABCPruner~\cite{LinJZZWT20}        &  0.94(4.35$\times$)  &                   -  & 70.29  & -      & 12+90
    \textbf{DNAL}($\lambda_a$=1e-4)    &  0.88(4.65$\times$)  &  7.18(3.56$\times$)  & 70.17  & 89.78  & 30+10+70
    \specialrule{0.10em}{0pt}{0pt}

3) MobileNetV2 on ImageNet-1K:

Comparison among several state-of-the-art methods for MobileNetV2 on ImageNet-1K

    Baseline                           &  300.79(1.00$\times$)  & 3.50(1.00$\times$)  & 71.52  & 90.15  & 120
    \specialrule{0.08em}{0pt}{0pt}
    \textbf{DNAL}($\lambda_a$=6e-5)    &  217.24(1.38$\times$)  &  2.87(1.22$\times$) & 71.02  & 89.96  & 80+10+90
    AMC~\cite{HeLLWLH18}               &  211(1.43$\times$)     &  -                  & 70.8   & -      & -     
    \textbf{DNAL}($\lambda_a$=7e-5)    &  207.25(1.45$\times$)  &  2.78(1.26$\times$) & 70.91  & 89.79  & 80+10+90
    \specialrule{0.10em}{0pt}{0pt}

4) ProxylessNAS on ImageNet-1K:

Comparison among several state-of-the-art methods for ProxylessNAS on ImageNet-1K

    \specialrule{0.10em}{0pt}{0pt}
    Baseline                            &  16.0   & 75.7   & 92.5   & -         & 150
    \specialrule{0.08em}{0pt}{0pt}
    \textbf{DNAL}($\lambda_a$=4e-5)     &   6.4   & -      & -      & gradient  & 100+10+110
    \textbf{DNAL}($\lambda_a$=5e-5)     &   4.4   & -      & -      & gradient  & 100+10+110
    \textbf{DNAL}($\lambda_a$=6e-5)     &   3.6   & 75.0   & 92.3   & gradient  & 100+10+110
    \specialrule{0.08em}{0pt}{0.5pt}
    \specialrule{0.08em}{0.5pt}{0pt}
    EA+BPE-1~\cite{ZhengJWYLTT20}       &   5.0   & 74.56  & -      & EA        & -
    CARS-A~\cite{YangWCSXXT19}          &   3.7   & 72.8   & 90.8   & EA        & -
    CARS-E~\cite{YangWCSXXT19}          &   4.4   & 73.7   & 90.8   & EA        & -
    \specialrule{0.08em}{0pt}{0pt}
    RL+BPE-1~\cite{ZhengJWYLTT20}       &   5.5   & 74.18  & -      & RL        & -
    NASNet-A~\cite{ZophVSL17}           &   5.3   & 74.0   & 91.6   & RL        & -
    NASNet-B~\cite{ZophVSL17}           &   5.3   & 72.8   & 91.3   & RL        & -
    NASNet-C~\cite{ZophVSL17}           &   4.9   & 72.5   & 91.0   & RL        & -
    MnasNet-92~\cite{TanCPVL18}         &   4.4   & 74.8   & -      & RL        & -
    MnasNet~\cite{TanCPVL18}            &   4.2   & 74.0   & -      & RL        & -
    MnasNet-65~\cite{TanCPVL18}         &   3.6   & 73.0   & -      & RL        & -
    \specialrule{0.08em}{0pt}{0pt}
    DARTS~\cite{LiuSY18}                &   4.7   & 73.3   & 91.3   & gradient  & 600\footnotemark[1]+250 
    ProxylessNAS~\cite{CaiZH18}         &   7.1   & 75.1   & 92.5   & gradient  & 200+150
    FBNet-A~\cite{WuDZWSWTVJ19}         &   4.3   & 73.0   & -      & gradient  & 90+360
    FBNet-B~\cite{WuDZWSWTVJ19}         &   4.5   & 74.1   & -      & gradient  & 90+360
    FBNet-C~\cite{WuDZWSWTVJ19}         &   5.5   & 74.9   & -      & gradient  & 90+360
    \specialrule{0.10em}{0pt}{0pt}

## Train for classification

>python train.py

## Authorship

This project is contributed by [Qingbei Guo](https://github.com/QingbeiGuo).

## Citation
