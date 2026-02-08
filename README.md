# VitalLM-25M: Biomedical Small Language Model 🧬

VitalLM is a **25M parameter** decoder-only Transformer model optimized for biomedical text generation. It was architected from scratch and trained on a **178M token** dataset synthesized from ChatDoctor (clinical dialogues) and PubMed abstracts.

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Status](https://img.shields.io/badge/Status-Trained-green)

## 🚀 Key Features
* **Custom Architecture:** Implements **SwiGLU** activation and **Learned Positional Embeddings** for optimized convergence on small scales.
* **Data Engineering:** Solved critical data distribution drift using **memory-mapped shuffling** on a 178M token corpus.
* **Performance:** Achieved **3.76 Validation Loss (Perplexity ~43)**, outperforming standard baselines for this parameter class.
* **Efficient Training:** Trained on a single NVIDIA P100 GPU using Mixed Precision (FP16).

## 📊 Training Metrics
| Metric | Value |
| :--- | :--- |
| **Parameters** | 25,050,000 |
| **Context Length** | 256 |
| **Vocab Size** | 16,384 (BPE) |
| **Final Val Loss** | 3.76 |
| **Training Steps** | 22,000 |

## 🛠️ Installation
```bash
git clone [https://github.com/YourUsername/VitalLM-25M.git](https://github.com/YourUsername/VitalLM-25M.git)
cd VitalLM-25M
pip install -r requirements.txt
