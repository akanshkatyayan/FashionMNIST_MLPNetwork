# FashionMNIST_MLPNetwork
Used MLP Neural Network with a STEM, Backbone and Classifier

## Problem Statement

Classify every image in terms of 1 out of the 10 classes


## Dataset

Title: MNIST Fashion Dataset
![image](https://user-images.githubusercontent.com/35501313/170054782-72c299ea-9602-4e69-a36d-251af208584a.png)

**Training Data:** 60,000 Images
**Test Data:** 10,000 Images

Used my_utils.py to load MNIST Fashion dataset which contains Gray scale images of different fashion items

Sources: (a) Origin: This dataset has been taken from the PyTorch using my_utils load_data_fashion_mnist function.

Relevant Information: Contains gray scale images of 10 labels as below:

* t-shirt
* trouser
* pullover
* dress
* coat
* sandal
* shirt
* sneaker
* bag
* ankle boot


## Model Development

### The STEM

![image](https://user-images.githubusercontent.com/35501313/170066284-d813f837-8c76-4566-8d13-d83860bbbae5.png)

* Patching the Images: Here, we are applying a patch function on an input image of 28X28 (total pixels 784) to transform it into smaller patches. On experimentation with different patch sizes of 4, 7, 2 for each image i.e. having 49(4x4 = 16 pixels) pacthes, 16 (7x7 = 49 pixels) patches, 196 (2x2 = 4 pixels) patches respectively. 49 patches of 4x4 siz eeach provided best results in terms of training time, test accuracy. Once, Image is patched into the required size, it is transformed into a verctor of dimension (49,16).
* Vector created using a patch function is then fed to a linear layer used from torch library. Input size for this Linear layer is the size of the patch i.e 16 and output size of the Linear layer is the size of the hidden layer. Linear layer is used to extract features from the input vector. After linear layer, Batchnormalization is applied on the output of the linear layer to normalize the hidden layer inputs.

### The Backbone

![image](https://user-images.githubusercontent.com/35501313/170066415-a83b6ca0-c54a-4dfa-af04-1c87f74e5dff.png)

Bachbone is the part of the model which is a combination of blocks of Hidden layers. Hidden layers are used to extract data (more features)from out input layer and provides the output to the output layer. More number of hidden layers are used to solve more complex problems. MLP (Multi-Layer perceptrons) architecture is used in the below model for implementing the hidden layers.

Below we are using 2 blocks of MLPs. Each MLP block has 2 Linear Layers. Linear layers are used to extract more features from the previous output. Input to the 1st block is the output of the stem block. Firstly, the matrix X is transposed and then fed to a Linear Layer 1. Output of the first linear layer is then activated using ReLU function to overcome vanishing gradient problem. Dropout layer is used to prevent overfitting on the output of the ReLU function. Finally, another Linear Layer is used to extract additional features before feeding it to MLP 2. Output from 1st MLP block O1 is first transposed and then fed to block MLP2 which is similar to MLP 1.

Similarly, Output of Block 1 is transposed and fed to Block 2 with. MLPS.

### Classifier

![image](https://user-images.githubusercontent.com/35501313/170066580-9dddf6b4-6351-42fd-b0b0-7044302d5cbf.png)

Classifier performs two operations: First, It takes the mean of the learned features from the image. torch.mean function has been used to take the mean of the output from the backbone and then fed to the final classifier layer. Classifier layer has the input size same as the output of the Backbone and output size of the number of classes i.e. 10

Finally, task is to apply the a classifier like softmax regression classifier. CrossEntropy loss function is used therefore, that willconvert the outputs to a probability distribution as in case of Softmax


## Training the Model

* The following function trains the model (net) on a training set (train_iter) for num_epochs.
* At the end of each epoch, the model is evaluated on a testing set (test_iter).
* Animator for visualizing the training progress.

## Result

![image](https://user-images.githubusercontent.com/35501313/170074397-f2e0e50e-115c-4b30-bfe6-133b48f1c588.png)


