# SkinLesNet: Classification of Skin Lesions and Detection of Melanoma Cancer Using a Novel Multi-Layer Deep Convolutional Neural Network

<p align="center">
  <a href="https://www.mdpi.com/2072-6694/16/1" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Cancers-2024-blue" alt="Journal"></a>
  <a href="https://doi.org/10.3390/cancers16010108" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/DOI-10.3390%2Fcancers16010108-red" alt="DOI"></a>
  <a href="https://github.com/azeemchaudharyg/SkinLesNet/notebooks/SkinLesNet_Project.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange" alt="Implementation"></a>
  <a href="LICENSE" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

Official repository accompanying the publication: **"SkinLesNet: Classification of Skin Lesions and Detection of Melanoma Cancer Using a Novel Multi-Layer Deep Convolutional Neural Network"** **Authors:** Muhammad Azeem, Khurram Kiani, Tarik Mansouri, and Nicholas Topping  

**Authors:** Muhammad Azeem, Kaveh Kiani, Taha Mansouri, and Nathan Topping

**Institution:** University of Salford, Manchester, England, UK.  

---

## Abstract

Skin cancer is a widespread disease that typically develops on the skin due to frequent exposure to sunlight. Although cancer can appear on any part of the human body, skin cancer accounts for a significant proportion of all new cancer diagnoses worldwide. There are substantial obstacles to the precise diagnosis and classification of skin lesions because of morphological variety and indistinguishable characteristics across skin malignancies. Recently, deep learning models have been used in the field of image-based skin-lesion diagnosis and have demonstrated diagnostic efficiency on par with that of dermatologists. To increase classification efficiency and accuracy for skin lesions, a cutting-edge multi-layer deep convolutional neural network termed SkinLesNet was built in this study. The dataset used in this study was extracted from the PAD-UFES-20 dataset and was augmented. The PAD-UFES-20-Modified dataset includes three common forms of skin lesions: seborrheic keratosis, nevus, and melanoma. To comprehensively assess SkinLesNet’s performance, its evaluation was expanded beyond the PAD-UFES-20-Modified dataset. Two additional datasets, HAM10000 and ISIC2017, were included, and SkinLesNet was compared to the widely used ResNet50 and VGG16 models. This broader evaluation confirmed SkinLesNet’s effectiveness, as it consistently outperformed both benchmarks across all datasets.

---

## Key Contributions
* **SkinLesNet Architecture:** Implements an advanced, multi-layer deep CNN architecture characterized by deep filter stacking, non-linear activation sequences, and structured pooling layers optimized for malignant pattern extraction.
* **Granular Lesion Categorization:** Moves beyond simple binary diagnostic setups to support highly sensitive, multi-class skin lesion categorization (including Melanoma, Basal Cell Carcinoma, and benign variations).
* **Robust Dermoscopic Assessment:** Validated across robust dermoscopic and smartphone-collected datasets, confirming highly stable diagnostic generalizability despite significant noise, illumination shifts, and imaging artifacts.
* **Optimized Clinical Footprint:** Engineers structural constraint balances to preserve micro-edge cellular boundaries while optimizing resource consumption for practical point-of-care deployment.

---

## Technical Overview: The SkinLesNet Model

The processing engine behind **SkinLesNet** transforms raw dermoscopic input matrices into discrete clinical classifications using a sequence of highly structured optimization stages:

1. **Input Map Acquisition:** Standardizing dermoscopic inputs to uniform dimensions to mitigate capture variations.
2. **Multi-Layer Convolutional Streams:** Iteratively stacking deep convolutional groups to map localized lesion edges, pigmentation boundaries, and global morphological contours.
3. **Non-Linear Operations & Batch Control:** Applying optimized ReLU functions and synchronized Batch Normalization tiers to sustain fast training convergences without experiencing overfitting or internal covariate shifts.
4. **Spatial Downsampling & Dense Classification:** Deploying MaxPooling matrices to aggregate feature hierarchies before passing the flattened structural vectors to fully connected dense layers and a softmax output classifier.

<p align="center">
  <img src="images/skinlesnet_architecture.png" alt="SkinLesNet Architecture Diagram" width="800"><br>
  <em>Figure: Layer configuration and algorithmic workflow of the proposed SkinLesNet model.</em>
</p>

---

## Installation & Setup

We recommend managing project environment variables using Anaconda. The repository is configured and validated for Python 3.7+ running TensorFlow with a Keras backend.

### 1. Create and Activate the Environment
```bash
conda create -n skinlesnet python=3.7 -y
conda activate skinlesnet
