# Relationship Classification CNN using NLP

This repository implements a Convolutional Neural Network (CNN) for classifying relationships between entities in text. The model is designed for the SemEval 2018 task and has been adapted for both English and Spanish (with a manually translated, revised, and corrected Spanish version). Additionally, a Weighted Gated Multimodal Unit (wGMU) variant is provided, along with a transfer learning implementation for a use case at the Cámara de Comercio de Medellín for the Colombian technology company Pratech.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation and Requirements](#installation-and-requirements)
- [Data](#data)
- [Models and Architectures](#models-and-architectures)
  - [CNN (Original)](#cnn-original)
  - [wGMU Implementation](#wgmu-implementation)
- [Training and Evaluation](#training-and-evaluation)
- [Transfer Learning and Pratech Use Case](#transfer-learning-and-pratech-use-case)
- [Experiments and Publication](#experiments-and-publication)
- [License](#license)
- [Contact](#contact)

## Overview

This project implements a CNN-based model for relationship classification between entities using NLP. It leverages embeddings generated with roBERTa-xml, and it has been evaluated on the SemEval 2018 dataset for both English and Spanish. The repository also includes an implementation using a Weighted Gated Multimodal Unit (wGMU) and a transfer learning adaptation for a real-world dataset from the Cámara de Comercio de Medellín.

## Repository Structure
```bash
Relationship-Classification-CNN-using-NLP/
│
├── Dataset/
│   ├── SemEval2018_EN_train.json
│   ├── SemEval2018_EN_test.json
│   ├── SemEval2018_ES_train.json   # Spanish version (translated and corrected)
│   └── SemEval2018_ES_test.json
│
├── Original/
│   ├── CNN.py          # Original CNN architecture for English and Spanish (10 classes)
│   ├── Dataset.py      # Functions for organizing and transforming data into embedding matrices
│   ├── Test.py         # Complete training and validation cycle (prints losses and accuracies; saves the trained model)
│   └── Rendimiento.py  # Loads the trained model and evaluates it on test data (accuracy, F1 score, confusion matrix)
│
├── wGMU/
│   ├── Dataset.py             # Adapted functions for the wGMU implementation
│   ├── Entrenamiento.py       # Training cycle adjusted for wGMU
│   ├── Red Neuronal.py        # Neural network architecture based on wGMU
│   ├── Rendimiento.py         # Evaluation of the wGMU model
│   └── wGMU modificada.py     # Modified version of the code for wGMU
│
├── TransferLearning/          # (Optional) Documentation and scripts for the Pratech use case
│   └── (Additional files and scripts)
│
└── README.md                  # This file
```

## Installation and Requirements

Ensure you have Python 3.7+ installed. The following packages are required:

- PyTorch
- Torchvision
- scikit-learn
- numpy
- tqdm

Install dependencies via pip:

```bash
pip install torch torchvision scikit-learn numpy tqdm
```


## Data

The **Dataset** folder contains JSON files for the SemEval 2018 task in both English and Spanish:
- **SemEval2018_EN**: Original English data.
- **SemEval2018_ES**: Translated, revised, and corrected Spanish data.

Raw text data is preprocessed and transformed into embeddings (using roBERTa-xml) prior to training.
## Models and Architectures

### CNN (Original)

The CNN architecture is defined in `Original/CNN.py`. It processes embeddings arranged in segments (e.g., pre-entity context, entities, inter-entity context) and outputs a classification among 10 classes, as defined in the SemEval task.
### wGMU Implementation

The **wGMU** folder contains an alternative implementation based on a Weighted Gated Multimodal Unit (wGMU). The files here (Dataset.py, Entrenamiento.py, Red Neuronal.py, Rendimiento.py, and wGMU modificada.py) mirror the structure of the original implementation but are adjusted for multimodal fusion.
## Training and Evaluation

### Training

Training is managed in the `Original/Test.py` script (or `wGMU/Entrenamiento.py` for the wGMU version). The process includes:
- **Data Splitting**: The provided training dataset is split into a training set and a validation set (e.g., 70% training, 30% validation) using `random_split` or similar methods.
- **Early Stopping**: Monitors validation loss to prevent overfitting.
- **Checkpointing**: The model state is saved during training.

#### Example Data Splitting (in the training script):

```python
# Define proportions
train_ratio = 0.9  # Data used for train+validation (from your provided training set)
# Compute sizes from full_dataset (which is already your training data)
train_size = int(0.7 * len(full_dataset))      # 70% for training
validation_size = len(full_dataset) - train_size # 30% for validation

# Split the dataset
train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])
```

## K-Fold Cross-Validation
For additional robustness, a 5-fold cross-validation is applied on the training data (excluding the reserved test set). This means that within your training dataset, you can split it into 5 folds and iterate:

- Use 4 folds for training.
- Use 1 fold for validation.
- Average metrics over the folds.

## Evaluation
After training, the final model is evaluated on the reserved test dataset using Original/Rendimiento.py (or wGMU/Rendimiento.py for the wGMU version). Metrics include accuracy, F1 score, and a confusion matrix.

## Transfer Learning and Pratech Use Case
A transfer learning adaptation has been implemented for a dataset from the Cámara de Comercio de Medellín. This use case, developed for Pratech, is available on Huggingface Spaces:

[Huggingface Space 1 (English Base)](https://huggingface.co/spaces/SantiagoMoreno-Col/NER_RC/tree/main/data/RC)

[Huggingface Space 2 (Spanish Use Case)](https://huggingface.co/spaces/joseamorenom02/NER_RC_ES)


## Experiments and Publication
A scientific paper documenting all experiments, tests, adjustments, and contributions is currently in preparation. Once published, it will be referenced in this repository. The paper includes:
- Comparisons between the original CNN architecture and the wGMU version.
- Experiments on both English and Spanish datasets.
- Transfer learning results for the Pratech use case.

## License
MIT License

## Contact
Author: José Moreno
GitHub: joseamorenom/Relationship-Classification-CNN-using-NLP
Email: joseamorenom02@gmail.com
