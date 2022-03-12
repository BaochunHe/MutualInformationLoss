# MutualInformationLoss
A pytorch implementation of mutual information loss for registration of two images based on the method of Mattes et al. The probability density distribution are estimated using  Parzen histograms rather than the simple gaussian distribution.
The Mutual information loss value was the same as the itkMattesMutualInformationImageToImageMetric. The loss cumputation was very fast and can be directly used in registration task.
