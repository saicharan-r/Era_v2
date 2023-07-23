# Assignment
1. Write a ResNet architecture for CIFAR10 that has the following architecture: 
    1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k] 
    2. Layer1 - 
        1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k] 
        2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
        3. Add(X, R1) 
    3. Layer 2 - 
        1. Conv 3x3 [256k] 
        2. MaxPooling2D 
        3. BN 
        4. ReLU 
    4. Layer 3 - 
        1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k] 
        2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k] 
        3. Add(X, R2) 
    5. MaxPooling with Kernel Size 4 
    6. FC Layer
    7. SoftMax 
2. Uses One Cycle Policy such that: 
    1. Total Epochs = 24 
    2. Max at Epoch = 5 
    3. LRMIN = FIND 
    4. LRMAX = FIND 
    5. NO Annihilation 
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8) 
4. Batch size = 512 
5. Use ADAM, and CrossEntropyLoss 
6. Target Accuracy: 90%

# Model

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 256, 16, 16]         294,912
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-19            [-1, 512, 4, 4]               0
      BatchNorm2d-20            [-1, 512, 4, 4]           1,024
             ReLU-21            [-1, 512, 4, 4]               0
           Conv2d-22            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
             ReLU-27            [-1, 512, 4, 4]               0
        MaxPool2d-28            [-1, 512, 1, 1]               0
           Linear-29                   [-1, 10]           5,130


# LR Finder
<img width="661" alt="Screenshot 2023-07-23 at 5 23 45 PM" src="https://github.com/saicharan-r/Erav1/assets/24753138/702ee3da-b1b1-4aab-8ef6-416de0c99ed3">

# Learning Rate
<img width="620" alt="Screenshot 2023-07-23 at 5 25 06 PM" src="https://github.com/saicharan-r/Erav1/assets/24753138/44edb274-ec20-4e8d-88db-a7e04ca4f137">

# Loss
<img width="921" alt="Screenshot 2023-07-23 at 5 26 12 PM" src="https://github.com/saicharan-r/Erav1/assets/24753138/46c10309-45c8-4bf0-87fe-dffecb17d364">

# Model Summary

Total params: 6,573,130  
Best training accuracy : 97.14  
Best test accuracy : 92.64
