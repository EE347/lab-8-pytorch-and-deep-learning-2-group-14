import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import confusion_matrix
import numpy as np

# Define the transform with a random horizontal flip and random rotation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),          # Horizontal flip with 50% probability
    transforms.RandomRotation(degrees=10)            # Random rotation up to 10 degrees
])

# Training function with confusion matrix saving
def train_model(model, criterion, optimizer, trainloader, testloader, device, epochs=4):
    # Initialize lists for recording losses
    train_losses = []
    test_losses = []
    best_train_loss = 1e9

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            # Apply the transforms to each image in the batch
            images = torch.stack([transform(image) for image in images])
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(trainloader))

        # Test loop
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Append predictions and labels for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_losses.append(test_loss / len(testloader))
        accuracy = correct / total

        # Generate and save the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix using matplotlib without seaborn
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        
        # Add annotations for each cell in the confusion matrix
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")

        # Save the confusion matrix, overwriting after each epoch
        plt.savefig('/home/pi/ee347/lab8/live_confusion_matrix.png')
        plt.close()

        print(f"Epoch: {epoch}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")

    return train_losses, test_losses

# Main Training Script
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Training with CrossEntropyLoss
    model_ce = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer_ce = optim.Adam(model_ce.parameters(), lr=0.001)
    criterion_ce = torch.nn.CrossEntropyLoss()
    print("Training with CrossEntropyLoss...")
    train_losses_ce, test_losses_ce = train_model(model_ce, criterion_ce, optimizer_ce, trainloader, testloader, device)

    # Training with NLLLoss (requires LogSoftmax layer)
    model_nll = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    model_nll = torch.nn.Sequential(model_nll, torch.nn.LogSoftmax(dim=1))  # Add LogSoftmax layer
    optimizer_nll = optim.Adam(model_nll.parameters(), lr=0.001)
    criterion_nll = torch.nn.NLLLoss()
    print("\nTraining with NLLLoss...")
    train_losses_nll, test_losses_nll = train_model(model_nll, criterion_nll, optimizer_nll, trainloader, testloader, device)

    # Plot the loss for comparison
    plt.figure()
    plt.plot(train_losses_ce, label="Train Loss (CrossEntropyLoss)")
    plt.plot(test_losses_ce, label="Test Loss (CrossEntropyLoss)")
    plt.plot(train_losses_nll, label="Train Loss (NLLLoss)")
    plt.plot(test_losses_nll, label="Test Loss (NLLLoss)")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training and Testing Loss for CrossEntropyLoss and NLLLoss")
    plt.savefig('/home/pi/ee347/lab8/loss_comparison_plot.png')
    plt.show()
