from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from app.cifar10_classifier import AssignmentCNN, get_device, get_transform


BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005
MODEL_PATH = Path("cnn_model.pth")


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(f"Using device: {device}")

    transform = get_transform()

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_number, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if batch_number % 100 == 0:
                accuracy = running_correct / running_total
                avg_loss = running_loss / batch_number
                print(
                    f"Epoch {epoch + 1}/{EPOCHS}, "
                    f"Batch {batch_number}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}"
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_correct / running_total
        print(
            f"Epoch {epoch + 1} complete. "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.4f}"
        )

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Accuracy on the 10000 test images: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
