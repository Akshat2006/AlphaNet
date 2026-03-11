# AlphaNet 🧠

A neural network built **from scratch in C++** to recognize handwritten letters (A-Z). No external ML libraries — just pure C++ with a custom linear algebra library.

## Architecture

```
Input (784) → Dense (128, ReLU) → Dense (64, ReLU) → Dense (26, Softmax)
```

- **Input**: 28×28 grayscale images flattened to 784 pixels
- **Hidden layers**: ReLU activation
- **Output**: 26 classes (A-Z) with Softmax + Cross-Entropy loss
- **Optimizer**: Stochastic Gradient Descent (SGD)

## Project Structure

```
AlphaNet/
├── mathlinalg.h/.cpp      # Custom VECTOR and MATRIX classes
├── activations.h/.cpp     # ReLU and Softmax activation functions
├── layers.h/.cpp          # Dense layer (forward pass)
├── loss.h/.cpp            # Cross-entropy loss and gradient
├── neuralnetwork.h/.cpp   # Full network (forward, backward, train, predict)
├── data_loader.h/.cpp     # EMNIST Letters CSV data pipeline
├── main.cpp               # Training loop with progress reporting
├── CMakeLists.txt         # CMake build configuration
├── download_data.py       # Dataset download script
└── generate_data.py       # Alternative data generation via emnist package
```

## Quick Start

### 1. Build
```bash
# Using g++
g++ -std=c++17 -O2 -o AlphaNet main.cpp mathlinalg.cpp activations.cpp layers.cpp loss.cpp neuralnetwork.cpp data_loader.cpp

# Or using CMake
mkdir build && cd build
cmake .. && cmake --build .
```

### 2. Get the Dataset
```bash
# Option A: Download EMNIST Letters from Kaggle
# https://www.kaggle.com/datasets/crawford/emnist
# Place emnist-letters-train.csv and emnist-letters-test.csv in data/

# Option B: Use the download script
python download_data.py

# Option C: Use the emnist Python package
pip install emnist
python generate_data.py
```

### 3. Train
```bash
./AlphaNet --epochs 10 --lr 0.001
```

### Command-Line Options
| Flag | Description | Default |
|------|-------------|---------|
| `--train <path>` | Training CSV path | `data/emnist-letters-train.csv` |
| `--test <path>` | Test CSV path | `data/emnist-letters-test.csv` |
| `--epochs <n>` | Number of training epochs | `5` |
| `--lr <rate>` | Learning rate | `0.001` |
| `--max-train <n>` | Limit training samples | all |
| `--max-test <n>` | Limit test samples | all |

## Dataset Format

CSV files with format:
```
label, pixel_0, pixel_1, ..., pixel_783
```
- Labels: 1-26 (A=1, B=2, ..., Z=26)
- Pixels: 0-255 grayscale values (normalized to 0-1 internally)

## License

MIT License
