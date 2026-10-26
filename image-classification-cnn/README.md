# 🖼️ Image Classification with a CNN

Classifies 32×32 color images into 10 everyday categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) with a small convolutional neural network built in PyTorch — a clean starter template for image classification.

## Dataset
**CIFAR-10** — 60,000 32×32 color images: 50,000 training and 10,000 test, across 10 classes.
Source: [toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html). Downloaded automatically on first run via `torchvision.datasets.CIFAR10` — no manual download needed.

## What it does
1. **Data** — loads CIFAR-10 with augmentation on the training split only (random crops + horizontal flips) and channel normalization
2. **Model** — small CNN: two conv blocks (32 → 64 channels) with max-pooling, then a 512-unit classifier head with dropout
3. **Training** — Adam optimizer, cross-entropy loss, learning-rate step decay; test accuracy tracked after every epoch
4. **Results** — training curves, a full 10×10 confusion matrix, and per-class accuracy (see `artifacts/` and `metrics.json`): **76.1% test accuracy after 5 epochs** (55.4% → 66.8% → 72.1% → 74.4% → 76.1%), best on ship (88.5%) and automobile (88.2%), hardest on cat (58.0%)
5. **Saved model** — trained weights in `model.pt`, ready to load and run predictions

## Tech
Python · PyTorch · torchvision · numpy · matplotlib

## Run it
```bash
pip install -r requirements.txt
python image_classification_cnn.py
```

Plots and `metrics.json` are written to `artifacts/`. For a quick smoke test on a small slice of the data:
```bash
python image_classification_cnn.py --epochs 1 --train-limit 2000
```
