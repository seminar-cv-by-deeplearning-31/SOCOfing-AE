---
title: SOCOfing autoencoders
---


# Introduction
In this blogpost we compress fingerprints from the Sokoto Coventry Fingerprint Dataset (SOCOfing) to a 1-Dimensional vector space using an [AutoEncoder (AE)] (https://pytorch-lightning-bolts.readthedocs.io/en/latest/autoencoders.html). Compressing fingerprints this way has multiple benefits. For one, it can reduce the size of databases drastically. Two, damaged fingerprints may be repaired through the Encoder Decoder process. Lastly, 1-D fingerprint representation may open up new possibilities in fingerprint matching, although we do not look at this problem in our blogpost.

Fingerprint grouping and classification, dactyloscopy, was invented at the end of the 19th century by Juan Vucetich, famously being used as evidence for a Argentinian Police case for the first time in 1892. Since then, the practice has been widely adopted for identification. Classically, the automation of dactyloscopy is done through a series of handcrafted feature extraction algorithms that perform roughly the same tasks as dactyloscopy by hand. 

Applying Deep Learning to fingerprint identification is nothing new. [TODO: cite ieee 7952518](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7952518) uses a Deep Convolutional Neural Net to extract high quality level features such as pores from fingerprints. [TODO: cite ieee 7852722](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7852722) uses a CNN to identify damaged fingerprints. [CITE 6996300](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6996300) uses a denoising AE to extract minutiae, features from fingerprints identified in the dactyloscopic process. More recently [cite: 9316670](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9316670) uses AE to detect anomelies in malicious fingerprint authentication attempts. To our knowledge, no one has actively tried to compress the representation of fingerprints using AE.

# Auto encoder architecture

[misschien mentionen dat een dense autoencoder niet werkte? Misschien ook niet want je had dat nog niet met de 32x32 fingerprints geprobeerd.]

After some failed attempts, we decided to keep our architecture choice simple, and use a standard autoencoder implemented by pytorch lightning bolt.

## [Section on resnet blocks as these are used in the autoencoder]



# Failed: 96x96 grayscale representation

In order to match fingerprints with a deterministic algorithm such as [https://github.com/kjanko/python-fingerprint-recognition] we need at least a full image of a fingerprint. We use the same ResNet-18 network as in previous experiments, increase the size of the input image and decrease the input channels to 1. We try two different loss functions: MSE-loss and BCE with logits with lr = 0.01, 0.001, 0.0001. In all 6 cases we do not find a valid reconstruction. See results. We suspect that the increased size of the image and also the additional white around the fingerprint make this a much harder problem than simply 32*32 square fingerprints. ~~Future works could look at increasing the network size or latent space, as well as different types of pooling and the addition of FC layers.~~

It seems that autoencoder architectures need to be changed considerably when scaling to larger image sizes. The encoder architecture we used consists of [4?] layers with each layer being a resnet block followed by a downsampling layer. After these [4?] layers, an adaptive average pool layer pools the feature maps to a vector number of 1x1 features.

When scaling up, we have tried using a larger resnet block in between each downsampling layer, which also results in a 4 times wider output vector. This however did not result in better reconstructions when trained for a similar amount of epochs as the resnet18 model. We hypothesize that the reconstructions might improve, when instead of using a larger resnet block, more downsampling sampling layers are used. Due to time constraints we were unable to test this.

We have also tried running the model for more epochs than the early stopping limit we chose. It is noteworthy that when we did this, the 96x96 fingerprints seemed to get more detailed, but the validation loss increased when the model trained for longer. The details that appeared in the reconstructions this way were also not correct when compared to the true images. A sample of this behaviour can be seen in figure \ref{fig overfitting 9696}.

\figure{fig overfitting 9696}
With early stopping

When continuing to train more epochs

[TODO: add true image]
\endfigure{}

Effects of different loss functions:

We first tried the MSE loss function. As we were not satisfied with the reconstructions, they looked blurry, while our fingerprint samples have sharp edges, we experimented with different loss functions. We tried MSE-loss, BCE loss, and L1-loss. In addition we also tried BCE + L1, and MSE + L1. The results of experimenting with different loss functions can be seen below:

[TODO: create plot with results]

Effect of the latent dimension

In this section we show our results of using different latent space sizes. This is at the core of our project, as we set out to find out how much fingerprints could be compressed using autoencoders. As we were unable to get the 96x96 fingerprints to work, we instead focused on the 32x32 center crops. We will show the effect of different latent space sizes qualitatively by showing reconstructions for each latent space size, as well as quantitatively by showing the average reconstruction losses we obtained.

|          |         8 |        16 |        32 |      64 |       128 |       256 |       512 |
|:---------|----------:|----------:|----------:|--------:|----------:|----------:|----------:|
| mse_loss | 0.0981035 | 0.0741007 | 0.0578159 | 0.04948 | 0.0467336 | 0.0484134 | 0.0503231 |

*Table1: Reconstruction losses for different latent dimensions with MSE-loss*

|          |         8 |        16 |        32 |      64 |       128 |       256 |       512 |
|:---------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| binary_cross_entropy_with_logits | 0.550273 | 0.505601 | 0.458729 | 0.469356 | 0.477316 | 0.470364 | 0.469483 |

*Table2: Reconstruction losses for different latent dimensions with BCE-loss*


Figure 1: MSE loss: reconstructions with latent dimensions 8, 16, 32, 64, 128, 256, 512. Rightmost column is true fingerprint

Figure 2: BCE-loss: reconstructions with latent dimensions 8, 16, 32, 64, 128, 256, 512. Rightmost column is true fingerprint

The figures and plots are shown for the epoch with the best validation loss in each run. Each run was done for a maximum of 500 epochs. The default resnet18 encoder block of the AutoEncoder model was used.

From the tables you can see that the lowest validation losses were obtained for a latent space of size 128 for mean squared error loss, and a latent space size of 64 for the binary cross entropy loss. For the MSE loss, the reconstructions look blurry when using a small latent space. When increasing the number of latent dimensions, the reconstructions increase sharpness. Some, e.g. the fourth row, seem to continue to increase in the amount of detail, but when comparing to the true fingerprints (rightmost column) it can be seen that these details are wrong. This is in line with the validation loss increasing again after 128 latent dimensions.

For the BCE loss, the run with 32 latent dimensions obtained the lowest validation loss. However, when looking at the reconstructions, they seem generally worse compared to the autoencoder trained with the MSE loss. Where the MSE loss has relatively consistent results, the BCE autoencoder seems to encode some fingerprints very well, while others are plagued by large patches of black.