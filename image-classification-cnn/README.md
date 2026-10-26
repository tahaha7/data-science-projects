# 🖼️ Image Classification with a CNN

Classifies 32×32 color images into 10 everyday categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) with a convolutional neural network built in PyTorch — a clean starter template for image classification.

## Dataset
**CIFAR-10** — 60,000 32×32 color images: 50,000 training and 10,000 test, across 10 classes.
Source: [toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html). Downloaded automatically on first run via `torchvision.datasets.CIFAR10` — no manual download needed.

## What it does
1. **Data** — loads CIFAR-10 with augmentation on the training split only (random crops + horizontal flips) and channel normalization
2. **Model (v2, current best)** — deeper CNN: three conv blocks (32 → 64 → 128 channels) with batch norm and max-pooling, then a 512-unit classifier head with dropout
3. **Training** — Adam optimizer, cross-entropy loss, learning-rate step decay every 5 epochs; test accuracy tracked after every epoch
4. **Results** — training curves, a full 10×10 confusion matrix, and per-class accuracy (see `artifacts_v2/` and `metrics.json`): **85.8% test accuracy after 15 epochs** (56.1% after epoch 1 → 85.8% after epoch 15), best on automobile (93.3%) and ship (93.2%), hardest on cat (73.9%, up from 58.0% in v1)
5. **Saved model** — trained weights in `model_v2.pt`, ready to load and run predictions
6. **Version 1** — the original small CNN (two conv blocks, 5 epochs) reached 76.1% test accuracy; script `image_classification_cnn.py`, weights `model.pt`, results in `artifacts/`

## Tech
Python · PyTorch · torchvision · numpy · matplotlib

## Run it
```bash
pip install -r requirements.txt
python image_classification_cnn_v2.py
```

Plots and `metrics.json` are written to `artifacts_v2/`. For a quick smoke test on a small slice of the data:
```bash
python image_classification_cnn_v2.py --epochs 1 --train-limit 2000
```
