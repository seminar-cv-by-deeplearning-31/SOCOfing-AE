---
title: SOCOfing autoencoders
---

# Introduction
In this blogpost we compress fingerprints from the Sokoto Coventry Fingerprint Dataset (SOCOfing) to a 1-Dimensional vector space using an [AutoEncoder (AE)](https://pytorch-lightning-bolts.readthedocs.io/en/latest/autoencoders.html). Compressing fingerprints this way has multiple benefits. For one, it could reduce the size of databases drastically. Imagine what a 50% compression rate would mean on a fingerprint database of the entire world population. Two, damaged fingerprints may be repaired through the Encoder Decoder process. Lastly, 1-D fingerprint representation may open up new possibilities in fingerprint matching based on 1-D vectors, although we do not look at this problem in our blogpost.

Fingerprint grouping and classification, dactyloscopy, was invented at the end of the 19th century by Juan Vucetich, famously being used as evidence for a Argentinian Police case for the first time in 1892. Since then, the practice has been widely adopted for identification. Classically, the automation of dactyloscopy is done through a series of handcrafted feature extraction algorithms that perform roughly the same tasks as dactyloscopy by hand. 

Applying Deep Learning to fingerprint identification is nothing new. [[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7952518) uses a Deep Convolutional Neural Net to extract high quality level features such as pores from fingerprints. [[2]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7852722) uses a CNN to identify damaged fingerprints. [[3]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6996300) uses a denoising AE to extract minutiae, features from fingerprints identified in the dactyloscopic process. More recently [[4]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9316670) uses AE to detect anomalies in malicious fingerprint authentication attempts. To our knowledge, no one has actively tried to compress the representation of fingerprints using AE.

# Autoencoder architecture

After some failed attempts, we decided to keep our architecture choice simple, and use a standard autoencoder implemented by [pytorch lightning bolt](https://pytorch-lightning-bolts.readthedocs.io/en/latest/autoencoders.html). This autoencoder makes use of resnet blocks, which is a commonly used architecture in deep learning. The autoencoder implementation provides two sizes of the resnet models. Most of the experiments use the resnet18 variant, as this variant needs a fraction of the compute of the resent50 variant. Some of the 96x96 input sizes were done with the resnet50 model.

We make two modifications to the lightning bolt Autoencoder. The number of image channels is changed from 3 to 1 to support the grayscale fingerprints as opposed to rgb images. In addition to that we override the training step to be able to change the loss function.

# Failed: 96x96 fingerprints

## Training

In order to match fingerprints with a deterministic algorithm such as [https://github.com/kjanko/python-fingerprint-recognition] we need at least a full image of a fingerprint. We use the same ResNet-18 network as in previous experiments, increase the size of the input image and decrease the input channels to 1. We try two different loss functions: MSE-loss and BCE with logits with lr = 0.01, 0.001, 0.0001. In all 6 cases we do not find a valid reconstruction. See results. We suspect that the increased size of the image and also the additional white around the fingerprint make this a much harder problem than simply 32*32 square fingerprints. 

It seems that autoencoder architectures need to be changed considerably when scaling to larger image sizes. The encoder architecture we used consists of 4 layers with each layer being a resnet block followed by a downsampling layer. After these 4 layers, an adaptive average pool layer pools the feature maps to a vector number of 1x1 features.

When scaling up, we have tried using a larger resnet block in between each downsampling layer, which also results in a 4 times wider output vector. This however did not result in better reconstructions when trained for a similar amount of epochs as the resnet18 model. We hypothesize that the reconstructions might improve, when instead of using a larger resnet block, more downsampling sampling layers are used. Due to time constraints we were unable to test this. This is something future work could explore.

We have also tried running the model for more epochs than the early stopping limit we chose. It is noteworthy that when we did this, the 96x96 fingerprints seemed to get more detailed, but the validation loss increased when the model trained for longer. The details that appeared in the reconstructions this way were also not correct when compared to the true images. A example of this behaviour can be seen in in the figures below:

<img height="320" src="https://user-images.githubusercontent.com/7264894/122401780-33b08480-cf7d-11eb-9ec7-a2cff91db3bb.png">

*Figure 1: Showing overfitting on 96x96 images. Columns show epoch 1, epoch with best validation loss, last epoch. Last column is the true fingerprint*

<img height="300" src="https://user-images.githubusercontent.com/7264894/122403774-ecc38e80-cf7e-11eb-9df3-df04b952435c.png">
<img height="300" src="https://user-images.githubusercontent.com/7264894/122403794-ee8d5200-cf7e-11eb-9493-ce11531d28e5.png">

Figure 2: Training and validation losses for a typical run on 96x96 images

These plots were created using the resnet50 model with 2048 latent dimensions. Similar behaviour was found for the resnet18 model, as well as with a smaller latent space of size 512.

## Matching
Even though we found that training the autoencoder on 96x96 images did not work, we tried to do fingerprint matching on the results. When using a deterministic dactyloscopy algorithm to match both overtrained samples and samples with least validation loss, the algorithm is unable to match the two fingerprints consistently. We use the algorithm found at (https://github.com/kjanko/python-fingerprint-recognition)[https://github.com/kjanko/python-fingerprint-recognition].

<img height="150" src="https://user-images.githubusercontent.com/7264894/122407111-a885bd80-cf81-11eb-89fd-812ccd3bf133.png">

*Matched pair of least validation loss generated sample and original. Even though a match is indicated, the match identifies the same minutiae at different locations.*

<img height="150" src="https://user-images.githubusercontent.com/7264894/122407149-b3405280-cf81-11eb-80b3-332f76772f43.png">

*This image shows the overtrained generated sample vs the original. Here, no match is found by the deterministic dactyloscopy algorithm.*


# Effects of different loss functions:

We first tried the MSE loss function. As we were not satisfied with the reconstructions, they looked blurry, while our fingerprint samples have sharp edges, we experimented with different loss functions. We tried MSE-loss, BCE loss, and L1-loss. In addition we also tried BCE + L1, and MSE + L1. The results of experimenting with different loss functions can be seen below:

![lossfunexp](https://user-images.githubusercontent.com/7264894/122392796-7f126500-cf74-11eb-86e7-e08634f8d1c1.png)

*Figure 3: reconstructions with: MSE-loss, MSE + L1 loss, BCE-loss, BCE+L1-loss, L1-loss. Last column is true fingerprint*

# Effect of number of latent dimensions

In this section we show our results of using different latent space sizes. This is at the core of our project, as we set out to find out how much fingerprints could be compressed using autoencoders. As we were unable to get the 96x96 fingerprints to work, we instead focused on the 32x32 center crops. We will show the effect of different latent space sizes qualitatively by showing reconstructions for each latent space size, as well as quantitatively by showing the average reconstruction losses we obtained.

|          |         8 |        16 |        32 |      64 |       128 |       256 |       512 |
|:---------|----------:|----------:|----------:|--------:|----------:|----------:|----------:|
| mse_loss | 0.0981035 | 0.0741007 | 0.0578159 | 0.04948 | 0.0467336 | 0.0484134 | 0.0503231 |

*Table 1: Reconstruction losses for different latent dimensions with MSE-loss*

|          |         8 |        16 |        32 |      64 |       128 |       256 |       512 |
|:---------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| binary_cross_entropy_with_logits | 0.550273 | 0.505601 | 0.458729 | 0.469356 | 0.477316 | 0.470364 | 0.469483 |

*Table 2: Reconstruction losses for different latent dimensions with BCE-loss*

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

It seems like the noisier the image the harder time the AE has to recreate the original, and somehow tries to return clean patterns. This pattern of Deep Learning models not liking noise has some supporting evidence found in scientific literature [[5]](https://arxiv.org/abs/1711.10925).

In the future we would like to find a way to make entire fingerprints reconstructable and testing to see if dactyloscopy can match recreations, or to see if a dactyloscopy expert can match patches. Additionally, seeing if different instances of the same fingerprint can be mapped to roughly the same 1-Dimensional vector representation.

# References

[1]: H. Su, K. Chen, W. J. Wong and S. Lai, "A deep learning approach towards pore extraction for high-resolution fingerprint recognition," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 2057-2061, doi: 10.1109/ICASSP.2017.7952518.

[2]: Y. Wang, Z. Wu and J. Zhang, "Damaged fingerprint classification by Deep Learning with fuzzy feature points," 2016 9th International Congress on Image and Signal Processing, BioMedical Engineering and Informatics (CISP-BMEI), 2016, pp. 280-285, doi: 10.1109/CISP-BMEI.2016.7852722.

[3]: A. Sankaran, P. Pandey, M. Vatsa and R. Singh, "On latent fingerprint minutiae extraction using stacked denoising sparse AutoEncoders," IEEE International Joint Conference on Biometrics, 2014, pp. 1-7, doi: 10.1109/BTAS.2014.6996300.

[4]: J. Kolberg, M. Grimmer, M. Gomez-Barrero and C. Busch, "Anomaly Detection With Convolutional Autoencoders for Fingerprint Presentation Attack Detection," in IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 3, no. 2, pp. 190-202, April 2021, doi: 10.1109/TBIOM.2021.3050036.

[5]: Ulyanov, Dmitry, et al, "Deep Image Prior," International Journal of Computer Vision, vol. 128, no. 7, July 2020, pp. 1867–88. arXiv.org, doi:10.1007/s11263-020-01303-4.
