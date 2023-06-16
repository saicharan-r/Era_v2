import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
from utils import GetCorrectPredCount
from matplotlib import pyplot as plt

class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        # Define the fully connected layers
        self.fc1 = nn.Linear(4096, 50, bias=False)
        self.fc2 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        # Define the forward pass of the network
        x = F.relu(self.conv1(x), 2)  # Apply convolution and activation
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Apply convolution, activation, and max pooling
        x = F.relu(self.conv3(x), 2)  # Apply convolution and activation
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # Apply convolution, activation, and max pooling
        x = x.view(-1, 4096)  # Reshape the tensor
        x = F.relu(self.fc1(x))  # Apply fully connected layer and activation
        x = self.fc2(x)  # Apply fully connected layer
        return F.log_softmax(x, dim=1)  # Apply softmax activation and return the output

    def train_step(self, model, device, train_loader, optimizer):
        model.train()
        pbar = tqdm(train_loader)  # Create a progress bar for training

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # Clear the gradients
            pred = model(data)  # Forward pass
            loss = F.nll_loss(pred, target)  # Calculate the loss
            train_loss += loss.item()
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            correct += GetCorrectPredCount(pred, target)  # Track the number of correct predictions
            processed += len(data)  # Track the number of processed samples

            # Update the progress bar description with loss and accuracy information
            pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        return 100*correct/processed, train_loss/len(train_loader)

    def test_step(self, model, device, test_loader):
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)  # Forward pass
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # Calculate the loss
                correct += GetCorrectPredCount(output, target)  # Track the number of correct predictions

        test_loss /= len(test_loader.dataset)  # Calculate the average loss

        # Print the test set results: average loss and accuracy
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

        return 100. * correct / len(test_loader.dataset), test_loss

    def run(self, num_epochs:int, model, device, train_loader, test_loader, optimizer, scheduler):
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        for epoch in range(1, num_epochs+1):
            print(f'Epoch {epoch}')
            # Training
            temp_acc, temp_loss = model.train_step(model, device, train_loader, optimizer)
            train_acc.append(temp_acc)
            train_losses.append(temp_loss)
            # Testing
            temp_acc, temp_loss = model.test_step(model, device, test_loader)
            test_acc.append(temp_acc)
            test_losses.append(temp_loss)
            scheduler.step()  # Update the learning rate using the scheduler
        return train_acc, train_losses, test_acc, test_losses
