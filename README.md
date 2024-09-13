# Character Recognition

## Description

This is a simple practice project in the field of computer vision, focusing on character recognition.
The project is in its first iteration, and its current state is, the code is able to recognize uppercase letters written on a light background without any noise.

## Implementation

The code uses a fixed dataset to train a multilayer perceptron. To do so, each image is binarized and inverted, and then the following list of descriptors is obtained:

- First 20 Fourier descriptors of the main contour, both real and imaginary parts separately
- First 20 Fourier descriptors of the main contour of the eroded version of the image, both real and imaginary parts separately
- Number of contours

The first descriptor is useful for encoding the shape of the letter. The second helps in cases where the width of the writing tool used is large, which might confuse the classifier. The last descriptor is very helpful in identifying inner contours, although it might cause problems when there is noise present.

When training the classifier, we made sure to reserve a 20% of the dataset to use for testing.

## Results

Considering the information above, the following results were achieved:

- Average accuracy: 72.9%
- Median accuracy: 73.4%

In order to obtain better results, we experimented with different descriptors and classifiers. Initially, only one type of descriptor was used, but adding more showed an improvement on the results. On the other hand, the following list of classifiers was tested: SVM, KNN, Naive Bayes, MLP. The last one provided the best results out of them all.
