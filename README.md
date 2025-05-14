# ActNet-HAR: Focus-Aware Multi-scale CNN for Human Action Recognition from Images

This repository provides the official TensorFlow implementation of **ActNet**, a novel deep learning architecture designed for human action recognition from still images. ActNet employs multi-scale convolutional processing and a hybrid attention mechanism to highlight the most informative regions of the image for accurate action classification.

---

## 🧠 Model Architecture

ActNet consists of the following components:
- ✅ **Multi-feature CNN backbone**
- ✅ **Activity Multi-scale Block (AMB)** – captures local and global features at multiple receptive fields
- ✅ **Focus-Aware Recognition Module (FARM)** – combines:
  - Self-attention for long-range spatial dependencies
  - Global spatial attention
  - Channel attention

<p align="center">
  <img src="images/architecture.png" alt="ActNet Architecture" width="600"/>
</p>

---

## 📄 Citation

> This work is currently under submission to a peer-reviewed journal.  
> Please check back later for an official DOI or arXiv link.

---

## 🗂️ Dataset

The model is designed and evaluated on:
- Stanford 40 Actions
- PASCAL VOC 2012

Please organize your dataset as follows:

dataset/
├── train/
│ ├── class_1/
│ ├── class_2/
│ └── ...
└── val/
├── class_1/
├── class_2/
└── ...


---

## 🚀 How to Train

1. 📦 Install the required packages:
```bash
pip install -r requirements.txt


python train.py


ActNet-HAR/
├── actnet_model.py         # Core model definition
├── train.py                # Training script
├── requirements.txt        # Dependency list
├── README.md               # Project summary
└── images/
    └── architecture.png    # (Optional) Architecture diagram




