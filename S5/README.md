## PART 1

### calculations
<img width="1419" alt="Screenshot 2023-06-10 at 7 07 53 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/15b983aa-d609-4ac7-986a-b33bd4d3954b">
<img width="1423" alt="Screenshot 2023-06-10 at 7 07 32 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/1dfd13cc-5980-443c-9ed8-791a459388e9">

### Model

<img width="413" alt="Screenshot 2023-06-10 at 7 08 24 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/8373a9fc-79f9-4258-ad33-d57a4b907e31">
The above model is a fully connected layer neural network with two inputs i1 and i2 and two outputs a_o1 and a_o2 with a hidden layer in between.
E_Total is the total loss calculated by a loss function.

### nodes and weights
<img width="165" alt="Screenshot 2023-06-10 at 7 19 25 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/87ca50ae-b18d-4672-ab6a-75c8cd5c91b0">
This image shows how the nodes and weights are connected by a mathematical relationship

### loss calculation and back propogation
<img width="165" alt="Screenshot 2023-06-10 at 7 19 25 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/36542e64-e4da-4d64-bcdb-232a049e8a9f">
Loss function is calculated as E_Total and the effect on loss function by each parameter in the neural network is found out using the partial derivative of the parameter with the loss function. In this image the loss function effect wrt to weight w5 is shown. As seen from the final value, w5 is dependent on previous layer outputs a_h1 and weights w1 and w2.
Similary the following table shows the derivates wrt other weights
<img width="294" alt="Screenshot 2023-06-10 at 7 27 25 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/a0a453c0-4a37-4af9-b652-748239f7db1a">
<img width="453" alt="Screenshot 2023-06-10 at 7 27 32 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/2a99d62e-25f0-4ff3-a7a4-81fd213a45f2">
<img width="304" alt="Screenshot 2023-06-10 at 7 29 07 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/587982d4-90ed-4bd0-9fd3-abacce9a0239">
<img width="462" alt="Screenshot 2023-06-10 at 7 29 32 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/7ff63d66-d4de-414c-a962-72491867e7c6">


And the table shows the calculations of these parameters in each iteration of back propogation


## learning rate and loss
As can be seen from loss reduces at a slower pace when learning rate is slow and at higher pace when learning rate is high. But this is true only for a certain slope. Smaller running rate will affect the time taken for the model to learn and higher learning rate will affect the accuracy of the model. Hence it is preferred to have higher learning rates in the initial iterations and smaller learning rates in the final iterations of the learning.

Below images shows the loss reduction with reference to each iteration of the learning for different learning rates

### 0.1
<img width="378" alt="Screenshot 2023-06-10 at 7 37 41 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/984a0b96-b8ad-4bc5-a343-2183842e2b0c">
### 0.2
<img width="386" alt="Screenshot 2023-06-10 at 7 37 56 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/2ef6c05c-3a46-4e92-b6ea-e5016156c4fd">
### 0.5
<img width="380" alt="Screenshot 2023-06-10 at 7 36 05 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/ea00b0fc-89b0-47ea-b81a-cd43661aefa6">
### 0.8
<img width="389" alt="Screenshot 2023-06-10 at 7 38 07 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/7d69a516-243f-450d-899b-a6a1a12cdf04">
### 1.0
<img width="376" alt="Screenshot 2023-06-10 at 7 38 16 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/794264f3-45ef-4caf-ac31-5aea951af749">
### 2.0
<img width="378" alt="Screenshot 2023-06-10 at 7 38 28 AM" src="https://github.com/saicharan-r/Erav1/assets/24753138/c64fd7c6-9560-45f3-8226-67478423378f">
