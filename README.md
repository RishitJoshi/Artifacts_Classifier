# Purpose
This Project aims to develop an classifier that can classify Archaeological artifacts based on images.Potentially such techniques  can help archaeologist in their assessment and classification of archaeological finds

# Method
The model is trained on Convulational Neural Network using VGG19 transfer learning plus the custom output layer on three different categories namely Basket, Coin and Figure 

# Dataset
The Dataset has been taken from sources like kaggle, http://collection-online.moa.ubc.ca/explore and contains 1000 images per class in trainig data and 100 images per class in validation data

# Model Performance 
This is the final model that yielded the highest accuracy:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
dense (Dense)                (None, 100)               2508900   
_________________________________________________________________
dropout (Dropout)            (None, 100)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 153       
=================================================================
Total params: 22,538,487
Trainable params: 2,514,103
Non-trainable params: 20,024,384

Training Accuracy:
![training](https://user-images.githubusercontent.com/47889475/98120608-2d7d0480-1ed4-11eb-8874-8b6b7d736ff2.JPG)

Evaluation(Test)Accuracy:
![test](https://user-images.githubusercontent.com/47889475/98120623-31108b80-1ed4-11eb-88df-66043eed9883.JPG)

Classification Metrics
![classification_metrics](https://user-images.githubusercontent.com/47889475/98120635-34a41280-1ed4-11eb-98c9-c01bc084365b.JPG)

Confusion Matrix
![Confusion_Matrix](https://user-images.githubusercontent.com/47889475/98120642-37066c80-1ed4-11eb-974c-e03311bb670f.JPG)




