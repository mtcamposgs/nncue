# ƎUƆИИ: NNCUE (Efficiently Updatable Complex Neural Networks)

**State-of-the-Art Neural Architecture for Computer Chess Engines**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📜 Abstract

Since the introduction of NNUE in 2018, computer chess has relied on shallow ReLU networks for evaluation. While fast, these networks suffer from limited expressivity. **NNCUE** proposes a novel architecture using **Residual Swish Gated Tension Units (ResSwiGTU)** to model chess dynamics as opposing tensions modulated by contextual intensity. 

Benchmarks show that NNCUE achieves **lower validation loss** (better understanding) and **competitive inference speed** compared to standard NNUE.

📄 **[Read the Full Paper (PDF)](./NNCUE_Paper.pdf)**

---

## 🧠 Architecture: ResSwiGTU Block

Unlike standard neurons that sum inputs, the Complex Neuron in NNCUE calculates a **Tension** ($L-R$) and a **Context Gate** ($L+R$).

$$ y = (L - R) \cdot \text{SiLU}(L + R) $$

This allows the network to capture non-linear relationships (pins, batteries, x-rays) efficiently.

### Visual Structure (v4.0)
1. **Input Stream**
2. **RMSNorm** (Stabilization)
3. **Linear Projection** (Split into Left/Right)
4. **SwiGTU** (Tension * SiLU(Context))
5. **Residual Connection** (Add to Input)

---

## 📊 Benchmarks

Tests performed on 12.9M chess positions (Self-play dataset).

| Model | Val Loss (MSE) | Throughput (pos/sec)* |
| :--- | :---: | :---: |
| Standard NNUE (ReLU) | 0.892 | 38,111 |
| **NNCUE v4.0 (SiLU)** | **0.841** | **46,200** |

*\*Speed measured in Batched CPU Inference (PyTorch). Real-world C++ engine implementation (AVX-512) is the next step.*

---

## 🚀 Usage

### 1. Requirements
```bash
pip install torch pandas numpy matplotlib
```

### 2. Running Benchmarks
Place a `positions.csv` (FEN, Evaluation) in the root folder and run:
```bash
python benchmark_ex.py
```

### 3. Using the Library
You can import the architecture into your own training pipeline:

```python
from nnc import NNCUE_Network

# Initialize model (compatible with HalfKP features)
model = NNCUE_Network_v4(num_inputs=768, hidden_dim=256, num_layers=2)

# Forward pass
score = model(input_indices, offsets)
```

---

## 🤝 Contribution

This project is a Proof of Concept for a new generation of Chess Evaluation Functions. We are looking for collaborators to:
- Port the **ResSwiGTU** block to C++ (Stockfish/Ethereal forks).
- Optimize SIMD/AVX-512 intrinsics for the RMSNorm + SiLU operations.

## 📝 Citation

If you use NNCUE in your research or engine, please cite:

```bibtex
@article{campos2025nncue,
  title={NNCUE: Efficiently Updatable Complex Neural Networks for Computer Chess},
  author={Campos, Matheus},
  year={2025},
  publisher={GitHub}
}
```

---
*Created by Matheus Campos, Independent Researcher.*
