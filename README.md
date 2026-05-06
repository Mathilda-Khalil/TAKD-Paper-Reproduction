# TAKD Paper Reproduction

This repository contains my reproduction and implementation work for the paper:

**Improved Knowledge Distillation via Teacher Assistant**  
Seyed Iman Mirzadeh et al., AAAI 2020.

---

## Project Overview

The goal of this project is to reproduce the main Teacher Assistant Knowledge Distillation (TAKD) experiments using CIFAR-10 and CIFAR-100 datasets with CNN and ResNet architectures.

The compared methods are:

- **NOKD:** student trained normally without distillation
- **BLKD:** direct knowledge distillation from teacher to student
- **TAKD:** teacher assistant distillation, where the teacher first trains an intermediate assistant model, then the assistant trains the student

---

## Architectures

### CNN
- Student (S): 2 layers
- Teacher Assistant (TA): 4 layers
- Teacher (T): 10 layers

### ResNet
- Student (S): ResNet8
- Teacher Assistant (TA): ResNet20
- Teacher (T): ResNet110

---

## Experimental Setup

- Dataset: CIFAR-10 and CIFAR-100
- Framework: PyTorch
- Platform: Google Colab
- Optimizer: SGD with momentum
- Learning Rate: 0.1
- Weight Decay: 0.0001

Some practical modifications were applied for reproduction purposes, including:
- reduced training epochs
- manual checkpoint saving
- manual evaluation scripts
- staged execution of NOKD, BLKD, and TAKD experiments

---

## Repository Files

- `train.py` → training pipeline
- `model_factory.py` → model creation utility
- `plain_cnn_cifar.py` → CNN architectures
- `resnet_cifar.py` → ResNet architectures
- `data_loader.py` → CIFAR data loading
- `TAKD_Colab_Implementation.py` → Colab implementation workflow

---

## Original Repository Credit

This implementation is based on the original public repository provided by the paper authors:

https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation

---

## Note

This repository is intended for academic reproduction and learning purposes.
