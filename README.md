# Project 2
# Machine Learning and Transportation 
## CarND-LeNet-Lab
### Team members：
### M105120312 王 磊
### M105120315 刘瀚中

## 1.Dataset:

In total, we used the MNIST data, we import data from ‘tensorflow.examples.tutorials.mnist’, which comes pre-loaded with TensorFlow. In Keras, the MNIST data that TensorFlow pre-loads comes as 28x28x1 images. But the LeNet architecture only accepts 32x32xC images, and in order to reformat the MNIST data from 28x28x1 into 32x32xC, a shape that LeNet will accept,  so we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).

## 2.Method:

### i. Data reading and parameter setting in Tensorflow :

After importing data, we used ‘dataset=mnist.load_data()’ to read datasets, because the data was already processed, so we set up ‘EPOCHS’ and ‘BATCH_SIZE’. In tensorflow1, we used two convolution layers, two pooling layers and three ‘Fully Connected’ layers. In tensorflow2, we used three convolution layers, three pooling layers and three ‘Fully Connected’ layers. In tensorflow3, we used two convolution layers, two pooling layers and four ‘Fully Connected’ layers. Then we create a training pipeline that uses the model to classify MNIST data, and evaluated how well the loss and accuracy of the model for a given dataset. After that, we run the training data through the training pipeline to train the model. After each epoch, we  measured the loss and accuracy of the validation set. Finally, we got ten group data of EPOCHs. 

### ii. Data reading and parameter setting in Keras :

After importing data, we used ‘dataset=mnist.load_data()’ to read datasets, first, we set ‘filters’, ‘kernel_size’, ‘strides’, and ‘input_shape’, and in keras1 and keras3, we used ’ Dropout(0.2)’ to dropout 30 percent neure kernel, and in keras2, we dropout 30 percent. Then we set pooling kernel. After that, we use three convolution layers in keras1 and keras2, but two convolution layers in keras3. After we ran the training data to train the model, we measured the loss and accuracy of the validation set and save the model after training. Finally we got ten group of data in each Keras. 

## 3.Accuracy comparison and analysis

### i. Tensorflow-1 VS Tensorflow-2

![t1-t2](https://github.com/WangLei-M105120312/project2/blob/main/image/t1-t2.png)  
From this line chart, we can see that the validation accuracy of tensorflow2 is significantly higher than that of tensorflow1. Therefore, adding a set of convolution layer and pooling layer can improve the accuracy of test data.

### ii. Tensorflow-1 VS Tensorflow-3

![t1-t3](https://github.com/WangLei-M105120312/project2/blob/main/image/t1-t3.png) 

It can be seen from the trend of discount in this figure that adding a full connection layer does not greatly improve the accuracy in the early epoch of training, or even slightly lower than before. With the continuation of training, the accuracy in the later epoches of training is improved compared with the previous tensorfow-1.

### iii. Keras-1 VS Keras-2

![k1-k2](https://github.com/WangLei-M105120312/project2/blob/main/image/k1-k2.png) 

Keras-2 changes the dropout value of keras-1 from 0.2 to 0.3. It can be seen from the final line chart of validation accuracy of each generation that when dropout value is 0.3, the validation accuracy is lower than 0.2.

In the training of neural network, we often encounter the problem of over fitting, which is shown in the following aspects: the loss function of the model in the training data is smaller, and the prediction accuracy is higher; but in the test data, the loss function is larger, and the prediction accuracy is lower. Dropout can effectively alleviate the occurrence of over fitting and achieve the effect of regularization to a certain extent.

### iv. Keras-1 VS Keras-2

![k1-k3](https://github.com/WangLei-M105120312/project2/blob/main/image/k1-k3.png)  
It is obvious from the figure that in this method, deleting a convolution layer has little effect on the validation accuracy, especially in the previous generations of training, the two almost coincide on the line chart.

### v.Test Accuracy of three Tensorflow solution

The three methods of tensorflow are compared. As shown in the table1, tensorflow-2 has the highest accuracy of 0.990.
|      Name     | tensorflow1 | tensorflow2 | tensorflow3 |
| :-----------: | :---------: | :---------: | :---------: | 
| Test Accuracy |    0.989    |     0.990   |    0.988    |


### vi.Test Accuracy of three Keras solution

The three methods of Keras are compared. As shown in the table2, Keras-3 has the highest accuracy of 0.97713.
|      Name     | Keras1 | Keras2 | Keras3 |
| :-----------: | :----: | :----: | :----: | 
| Test Accuracy | 0.9892 | 0.9886 | 0.9913 |

## 4.Summary
