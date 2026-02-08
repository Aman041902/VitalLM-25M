# VitalLM-25M: Biomedical Small Language Model 🧬

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Status](https://img.shields.io/badge/Status-Trained-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Weights_Available-yellow)

**VitalLM-25M** is a custom-architected, decoder-only Transformer model optimized for biomedical text generation. Architected from scratch and trained on a **178M token** dataset synthesized from clinical dialogues (ChatDoctor) and medical literature (PubMed), it demonstrates that small, specialized models can achieve robust reasoning without billion-scale parameters.

## 🚀 Key Features
* **Custom Architecture:** Implements **SwiGLU** activation for improved geometric coherence and **Learned Positional Embeddings**.
* **Data Engineering:** Solved critical data distribution drift (Validation Loss 6.0 → 3.76) using **memory-mapped shuffling** on a 178M token corpus.
* **Performance:** Achieved **3.76 Validation Loss (Perplexity ~43)**, outperforming standard baselines for this parameter class.
* **Efficient Training:** Trained on a single NVIDIA P100 GPU using Mixed Precision (FP16) and Gradient Clipping.

---

## 📊 Training Performance
The model was trained for **22,000 iterations** (approx 1 epoch) on a mixed corpus.

### 1. Loss Convergence
*The model shows steady convergence with no signs of overfitting, thanks to the memory-mapped shuffling strategy.*
![Loss Curve](assets/loss_curve.png)

### 2. Perplexity (Intelligence Score)
*Perplexity dropped from ~427 to ~43, indicating a 10x improvement in prediction confidence.*
![Perplexity](assets/perplexity.png)

---

## 🛠️ Technical Specifications
| Metric | Value |
| :--- | :--- |
| **Parameters** | 25,050,000 |
| **Layers** | 12 |
| **Heads** | 8 |
| **Embedding Dim** | 320 |
| **Context Length** | 256 |
| **Vocab Size** | 16,384 (Custom BPE) |
| **Final Val Loss** | 3.76 |

---

## 🧠 Quick Start
You can load the model directly from Hugging Face using the custom `model.py` architecture included in this repo.

### 1. Installation
```bash
git clone [https://github.com/YourUsername/VitalLM-25M.git](https://github.com/Aman041902/VitalLM-25M.git)
cd VitalLM-25M
pip install torch transformers tokenizers huggingface_hub
