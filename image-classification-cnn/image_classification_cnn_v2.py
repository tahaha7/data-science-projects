# Image classification on CIFAR-10, version 2: a deeper CNN, trained longer.
# Same data and setup as v1, but three conv blocks instead of two,
# batch norm to keep the deeper net training well, and 15 epochs.

import argparse
import json
import os
import random
import time

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
ARTIFACTS = os.path.join(HERE, "artifacts_v2")


# Deeper CNN: three blocks (32 -> 64 -> 128 channels), batch norm each block.
class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.drop = nn.Dropout(0.25)

    def _block(self, x, conv_a, bn_a, conv_b, bn_b):
        x = F.relu(bn_a(conv_a(x)))
        x = F.relu(bn_b(conv_b(x)))
        return self.pool(x)

    def forward(self, x):
        x = self._block(x, self.conv1, self.bn1, self.conv2, self.bn2)
        x = self._block(x, self.conv3, self.bn3, self.conv4, self.bn4)
        x = self._block(x, self.conv5, self.bn5, self.conv6, self.bn6)
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
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                conf[t, p] += 1
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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--resume", action="store_true",
                    help="pick up from checkpoint_v2.pt if it exists")
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

    model = DeeperCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_acc": []}
    start_epoch = 0
    ckpt_path = os.path.join(HERE, "checkpoint_v2.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        print(f"Resuming from epoch {start_epoch} (checkpoint had "
              f"{len(history['test_acc'])} epochs)", flush=True)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
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
              f"test accuracy {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        # Save progress every epoch, so a restart never loses the run.
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                    "history": history, "classes": CLASSES,
                    "sched": sched.state_dict(),
                    "opt": opt.state_dict()}, ckpt_path)

    acc, conf, per_class = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {acc:.4f}")
    print("Per-class accuracy:", per_class)

    torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                "test_accuracy": acc}, os.path.join(HERE, "model_v2.pt"))
    plot_curves(history, os.path.join(ARTIFACTS, "training_curves.png"))
    plot_confusion(conf, os.path.join(ARTIFACTS, "confusion_matrix.png"))
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": round(acc, 4), "per_class_accuracy": per_class,
                   "train_loss": history["train_loss"], "test_acc": history["test_acc"],
                   "epochs": args.epochs, "train_size": len(train)}, f, indent=2)
    print("Saved model_v2.pt and plots in artifacts_v2/")


if __name__ == "__main__":
    main()
