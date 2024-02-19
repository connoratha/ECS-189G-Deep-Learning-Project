
# Ensure all necessary imports are present
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Transformations
transform_cifar = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CIFAR-10 Data Loaders
train_dataset_cifar = torchvision.datasets.CIFAR10(root='./Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/stage_3_data-3/CIFAR', train=True, download=True, transform=transform_cifar)
test_dataset_cifar = torchvision.datasets.CIFAR10(root='./Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/stage_3_data-3/CIFAR', train=False, download=True, transform=transform_cifar)
train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=batch_size, shuffle=True)
test_loader_cifar = DataLoader(test_dataset_cifar, batch_size=batch_size, shuffle=False)

# MNIST Data Loaders
train_dataset_mnist = torchvision.datasets.MNIST(root='./Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/stage_3_data-3/CIFAR', train=True, download=True, transform=transform_mnist)
test_dataset_mnist = torchvision.datasets.MNIST(root='./Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/Project code/stage_3_data-3/CIFAR', train=False, download=True, transform=transform_mnist)
train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=batch_size, shuffle=True)
test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=batch_size, shuffle=False)

# Improved CNN Architecture
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10, init_channels=3):
        super(ImprovedCNN, self).__init__()
        self.init_channels = init_channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(init_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Dynamically calculate the size of the flattened features for the first fully connected layer
        if init_channels == 3:  # Assuming CIFAR-10
            self.flat_features = 64 * 8 * 8
        else:  # Assuming MNIST
            self.flat_features = 64 * 7 * 7

        self.fc1 = nn.Linear(self.flat_features, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, self.flat_features)  # Adjust flattening to the dynamic size
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# Function to train and evaluate the model
# Function to train and evaluate the model

def train_and_evaluate(model, train_loader, test_loader, num_epochs, dataset_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # List to store average loss per epoch for plotting
    losses = []

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'{dataset_name} - Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

    # Plotting the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, label=f'Training Loss - {dataset_name}')
    plt.title(f'Training Loss Convergence - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluation loop
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions) * 100  # Convert to percentage
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Accuracy of the {dataset_name} model on the test images: {accuracy:.2f}%')
    print(f'Precision of the {dataset_name} model on the test images: {precision:.2f}')
    print(f'Recall of the {dataset_name} model on the test images: {recall:.2f}')
    print(f'F1 Score of the {dataset_name} model on the test images: {f1:.2f}')



# Train and evaluate on CIFAR-10
print("Training and evaluating on CIFAR-10 dataset...")
cifar_model = ImprovedCNN(num_classes=10, init_channels=3).to(device)
train_and_evaluate(cifar_model, train_loader_cifar, test_loader_cifar, num_epochs, "CIFAR-10")

# Train and evaluate on MNIST
print("\nTraining and evaluating on MNIST dataset...")
mnist_model = ImprovedCNN(num_classes=10, init_channels=1).to(device)
train_and_evaluate(mnist_model, train_loader_mnist, test_loader_mnist, num_epochs, "MNIST")




