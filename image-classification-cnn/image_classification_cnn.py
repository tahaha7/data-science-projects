# Image classification on CIFAR-10 with a small CNN in PyTorch.
# Downloads the data itself, trains, and saves the model + plots.

import argparse
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")


# Small CNN, nothing fancy. Two conv blocks, then a classifier head.
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def get_data(data_dir, train_limit=None):
    # Train gets augmentation, test stays clean.
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    if train_limit:  # quick smoke run, not the real result
        train = Subset(train, range(train_limit))
    return train, test


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    conf = np.zeros((10, 10), dtype=int)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                conf[t, p] += 1
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    # Per-class accuracy, straight from the confusion matrix diagonal.
    per_class = {CLASSES[i]: round(float(conf[i, i] / max(conf[i].sum(), 1)), 4)
                 for i in range(10)}
    return correct / total, conf, per_class


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, history["train_loss"], marker="o")
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("Epoch")
    ax[1].plot(epochs, history["test_acc"], marker="o")
    ax[1].set_title("Test accuracy")
    ax[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(conf, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, conf[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    args = ap.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    os.makedirs(ARTIFACTS, exist_ok=True)

    train, test = get_data(args.data_dir, args.train_limit)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)
    print(f"Train images: {len(train)}, test images: {len(test)}")

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_acc": []}
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        acc, _, _ = evaluate(model, test_loader, device)
        history["train_loss"].append(round(running / len(train), 4))
        history["test_acc"].append(round(acc, 4))
        print(f"Epoch {epoch + 1}/{args.epochs}: loss {history['train_loss'][-1]:.4f}, "
              f"test accuracy {acc:.4f}", flush=True)

    acc, conf, per_class = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {acc:.4f}")
    print("Per-class accuracy:", per_class)

    torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                "test_accuracy": acc}, os.path.join(HERE, "model.pt"))
    plot_curves(history, os.path.join(ARTIFACTS, "training_curves.png"))
    plot_confusion(conf, os.path.join(ARTIFACTS, "confusion_matrix.png"))
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": round(acc, 4), "per_class_accuracy": per_class,
                   "train_loss": history["train_loss"], "test_acc": history["test_acc"],
                   "epochs": args.epochs, "train_size": len(train)}, f, indent=2)
    print("Saved model.pt and plots in artifacts/")


if __name__ == "__main__":
    main()
