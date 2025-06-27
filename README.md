# 🔧 Execution Environment and Instructions

All experiments in this project were conducted using **Python** and **PyTorch**.

The core architecture implements the **Residual Flow model** as proposed by *Chen et al. (2021)*, enabling exact likelihood estimation via a chain of **invertible residual blocks**.  
All experiments were executed on **CUDA-compatible GPUs** for efficient training.

---

## 🛠️ Environment Setup

To ensure reproducibility, we provide a **Conda environment specification** (`residualflow_env.yml`) that lists all required dependencies.

You can create and activate the environment using the following commands:

```bash
conda env create -f residualflow_env.yml
conda activate residualflow
```

This environment includes:

- `Python (≥3.8)`
- `PyTorch (≥1.10)`
- `torchvision`
- `matplotlib`
- `numpy`
- `tqdm`
- `imageio`
- `PyYAML`

These packages are essential for training, data preprocessing, visualization, and configuration management.

---

## 📦 Codebase Structure

The code is modular and organized into the following key components:

- **`main.py`**  
  Entry point for training and evaluation. Loads the configuration, initializes the model, and launches the training process.

- **`train.py`**  
  Implements the training loop, loss computation, and model evaluation routines.

- **`residualflow.py`**  
  Constructs the overall model by composing a sequence of invertible transformations.

- **`iresblock.py`**  
  Defines the invertible residual blocks used in the model. These blocks employ spectral normalization to enforce Lipschitz continuity.

- **`container.py`**  
  Provides a wrapper for chaining multiple invertible modules with consistent forward and inverse interfaces.

- **`normalization.py`**  
  Contains normalization layers such as Moving BatchNorm, adapted to maintain invertibility.

- **`elemwise.py`**  
  Implements channel-wise affine transformations such as zero-mean shifting and scaling.

- **`visualize_flow.py`**  
  Offers tools for visualizing the learned flow transformations and grid deformations.

---

## 🚀 Running the Model

To begin training the model (e.g., on CIFAR-10), simply run:

```bash
python main.py
```

---

## 📊 Training Details

The training process:

- Automatically logs relevant metrics  
- Saves checkpoints to the default output directory  
- Allows configuration of hyperparameters (learning rate, batch size, number of residual blocks, optimizer settings) via `main.py` or direct code editing  
- Uses PyTorch’s serialization functions to save the model, which can be restored for evaluation or fine-tuning






