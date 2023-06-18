# Iteration 1

## Target
The target is to setup the model and get the skeleton right

## Results
Parameters = 194,884
Best training accuracy = 99.53%
Best testing accuracy = 99.01%
Epochs - 20

## Analysis
It is a big model with close 200k parameters
Model is overfitting as the training accuracy is more and test accuracy is less

https://github.com/saicharan-r/Erav1/blob/main/S7/S7-1.ipynb


# Iteration 2

## Target
To reduce the parameters

## Results
Parameters = 9,990
Best training accuracy = 99.03%
Best testing accuracy = 98.84%
Epochs - 20

## Analysis
Model has around 10k parameters which is lesser than the target model but closer to it
The gap between the test and training accuracy is reduced meaning the model has trained in a right way but the efficiency is to be improved

https://github.com/saicharan-r/Erav1/blob/main/S7/S7-2.ipynb



# Iteration 3

## Target
Achieved the desired target through augmentation and regularisation of input data and using GAP to reduce the params in the last layer

# Results
Parameters = 7,750
Best training accuracy = 99.12
Best testing accuracy = 99.46
Epochs - 15


# Analysis
Model is trained well and with less parameters and is within the limits.
The image augmentation and regularisation along with the plateau learning rate helped to made the model efficient
Achieved consistent 99.4% accuracy in the last few runs


https://github.com/saicharan-r/Erav1/blob/main/S7/S7-3.ipynb


