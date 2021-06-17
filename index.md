---
title: SOCOfing autoencoders
---

Note: work in progress

# Introduction
In this blogpost we compress fingerprints from the Sokoto Coventry Fingerprint Dataset (SOCOfing) to a 1-Dimensional vector space using an [AutoEncoder (AE)](https://pytorch-lightning-bolts.readthedocs.io/en/latest/autoencoders.html). Compressing fingerprints this way has multiple benefits. For one, it canmay reduce the size of databases drastically. Imagine what a 50% compression rate would mean on a fingerprint database of the entire world population. Two, damaged fingerprints may be repaired through the Encoder Decoder process. Lastly, 1-D fingerprint representation may open up new possibilities in fingerprint matching based on 1-D vectors, although we do not look at this problem in our blogpost.

Fingerprint grouping and classification, dactyloscopy, was invented at the end of the 19th century by Juan Vucetich, famously being used as evidence for a Argentinian Police case for the first time in 1892. Since then, the practice has been widely adopted for identification. Classically, the automation of dactyloscopy is done through a series of handcrafted feature extraction algorithms that perform roughly the same tasks as dactyloscopy by hand. 

Applying Deep Learning to fingerprint identification is nothing new. [TODO: cite ieee 7952518](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7952518) uses a Deep Convolutional Neural Net to extract high quality level features such as pores from fingerprints. [TODO: cite ieee 7852722](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7852722) uses a CNN to identify damaged fingerprints. [CITE 6996300](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6996300) uses a denoising AE to extract minutiae, features from fingerprints identified in the dactyloscopic process. More recently [cite: 9316670](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9316670) uses AE to detect anomelies in malicious fingerprint authentication attempts. To our knowledge, no one has actively tried to compress the representation of fingerprints using AE.

# Auto encoder architecture

After some failed attempts, we decided to keep our architecture choice simple, and use a standard autoencoder implemented by pytorch lightning bolt.

## Resnet blocks

The autoencoder uses resnet blocks in both the en-and decoder.

TODO: Write short section on resnets



# Failed: 96x96 grayscale representation

In order to match fingerprints with a deterministic algorithm such as [https://github.com/kjanko/python-fingerprint-recognition] we need at least a full image of a fingerprint. We use the same ResNet-18 network as in previous experiments, increase the size of the input image and decrease the input channels to 1. We try two different loss functions: MSE-loss and BCE with logits with lr = 0.01, 0.001, 0.0001. In all 6 cases we do not find a valid reconstruction. See results. We suspect that the increased size of the image and also the additional white around the fingerprint make this a much harder problem than simply 32*32 square fingerprints. ~~Future works could look at increasing the network size or latent space, as well as different types of pooling and the addition of FC layers.~~

It seems that autoencoder architectures need to be changed considerably when scaling to larger image sizes. The encoder architecture we used consists of [4?] layers with each layer being a resnet block followed by a downsampling layer. After these [4?] layers, an adaptive average pool layer pools the feature maps to a vector number of 1x1 features.

When scaling up, we have tried using a larger resnet block in between each downsampling layer, which also results in a 4 times wider output vector. This however did not result in better reconstructions when trained for a similar amount of epochs as the resnet18 model. We hypothesize that the reconstructions might improve, when instead of using a larger resnet block, more downsampling sampling layers are used. Due to time constraints we were unable to test this.

We have also tried running the model for more epochs than the early stopping limit we chose. It is noteworthy that when we did this, the 96x96 fingerprints seemed to get more detailed, but the validation loss increased when the model trained for longer. The details that appeared in the reconstructions this way were also not correct when compared to the true images. A sample of this behaviour can be seen in in the figure below:

*![2048_epochs_with_first](https://user-images.githubusercontent.com/7264894/122401780-33b08480-cf7d-11eb-9ec7-a2cff91db3bb.png)

Figure 1: Showing overfitting on 96x96 images. Columns show epoch 1, best epoch w.r.t validation loss, last epoch. Last column is the true fingerprint*

![2048_train_loss](https://user-images.githubusercontent.com/7264894/122403774-ecc38e80-cf7e-11eb-9df3-df04b952435c.png)
![W B Chart 17_06_2021, 15_14_55](https://user-images.githubusercontent.com/7264894/122403794-ee8d5200-cf7e-11eb-9493-ce11531d28e5.png)

Figure 2: Plot of training and validation losses for 96x96 images

# Effects of different loss functions:

We first tried the MSE loss function. As we were not satisfied with the reconstructions, they looked blurry, while our fingerprint samples have sharp edges, we experimented with different loss functions. We tried MSE-loss, BCE loss, and L1-loss. In addition we also tried BCE + L1, and MSE + L1. The results of experimenting with different loss functions can be seen below:

![lossfunexp](https://user-images.githubusercontent.com/7264894/122392796-7f126500-cf74-11eb-86e7-e08634f8d1c1.png)

*Figure 3: reconstructions with: MSE-loss, MSE + L1 loss, BCE-loss, BCE+L1-loss, L1-loss. Last column is true fingerprint*

# Effect of number of latent dimensions

In this section we show our results of using different latent space sizes. This is at the core of our project, as we set out to find out how much fingerprints could be compressed using autoencoders. As we were unable to get the 96x96 fingerprints to work, we instead focused on the 32x32 center crops. We will show the effect of different latent space sizes qualitatively by showing reconstructions for each latent space size, as well as quantitatively by showing the average reconstruction losses we obtained.

|          |         8 |        16 |        32 |      64 |       128 |       256 |       512 |
|:---------|----------:|----------:|----------:|--------:|----------:|----------:|----------:|
| mse_loss | 0.0981035 | 0.0741007 | 0.0578159 | 0.04948 | 0.0467336 | 0.0484134 | 0.0503231 |

*Table1: Reconstruction losses for different latent dimensions with MSE-loss*

|          |         8 |        16 |        32 |      64 |       128 |       256 |       512 |
|:---------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| binary_cross_entropy_with_logits | 0.550273 | 0.505601 | 0.458729 | 0.469356 | 0.477316 | 0.470364 | 0.469483 |

*Table2: Reconstruction losses for different latent dimensions with BCE-loss*

![latdim_mse](https://user-images.githubusercontent.com/7264894/122271731-082b8c80-cee0-11eb-9790-6e891192c3a6.png "latent dim: mse loss")

*Figure 4: MSE loss: reconstructions with latent dimensions 8, 16, 32, 64, 128, 256, 512. Rightmost column is true fingerprint*

![latdim_bce](https://user-images.githubusercontent.com/7264894/122271812-209ba700-cee0-11eb-9c60-e7a892c70e87.png)

*Figure 5: BCE-loss: reconstructions with latent dimensions 8, 16, 32, 64, 128, 256, 512. Rightmost column is true fingerprint*

The figures and plots are shown for the epoch with the best validation loss in each run. Each run was done for a maximum of 500 epochs. The default resnet18 encoder block of the AutoEncoder model was used.

From the tables you can see that the lowest validation losses were obtained for a latent space of size 128 for mean squared error loss, and a latent space size of 64 for the binary cross entropy loss. For the MSE loss, the reconstructions look blurry when using a small latent space. When increasing the number of latent dimensions, the reconstructions increase sharpness. Some, e.g. the fourth row, seem to continue to increase in the amount of detail, but when comparing to the true fingerprints (rightmost column) it can be seen that these details are wrong. This is in line with the validation loss increasing again after 128 latent dimensions.

For the BCE loss, the run with 32 latent dimensions obtained the lowest validation loss. However, when looking at the reconstructions, they seem generally worse compared to the autoencoder trained with the MSE loss. Where the MSE loss has relatively consistent results, the BCE autoencoder seems to encode some fingerprints very well, while others are plagued by large patches of black.

# Discussion and Conclusion

Our original goal was to see if fingerprints can be compressed and how far. However, because we were unable to create a converging model for entire fingerprints (96*96), we were unable to test if a deterministic dactyloscopic algorithm could match the recreated fingerprints to their original version. We can however inspect the 32*32 patches with our limited knowledge of dactyloscopy and identify at least some rough patterns. Firstly, we see that even though some reconstructions come close, details are not exactly matching. We have some concern for saving fingerprints this way and using it for identification, as reconstructions are only approximations of real fingerprints. 

Second, it seems that the AE network tries very hard to restore noisy images. See Figure:

<img width="66" src="https://user-images.githubusercontent.com/7264894/122272345-bc2d1780-cee0-11eb-8724-c4eec1a02200.png">

*Left: Reconstructed, Right: Original*

It seems like the noisier the image the harder time the AE has to recreate the original, and somehow tries to return clean patterns. This pattern of Deep Learning models not liking noise has some supporting evidence found in scientific literature. [https://arxiv.org/abs/1711.10925]

In the future we would like to find a way to make entire fingerprints reconstructable and testing to see if dactyloscopy can match recreations, or to see if a dactyloscopy expert can match patches. Additionally, seeing if different instances of the same fingerprint can be mapped to roughly the same 1-Dimensional vector representation.
