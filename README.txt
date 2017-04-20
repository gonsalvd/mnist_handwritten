Project 2 - EEL6935 (Deep Learning)

MNIST Handwritten Dataset

Mod vs. Orig - 

Original dataset contains 28x28 sample images. readMNIST() gets ride of padding
and downsizes to 20x20.

Mod dataset uses the 20x20 downsized images and 'scales/stretches' digits to fill
the given square.

PCA - 

PCA is done on both orig/mod image sets. PCA is done a random 10k subset of training.
Also tested taking random subset of 500,1000 and look at variability in the sample
means (mu) output from pca(). Seem less variable at a larger random sample.

PCA (Variance) - 

More variance was explained in less samples with the modified image sets.