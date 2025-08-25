# Advanced Breast Cancer Segmentation with DSU-Net

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

### **Project Output: Model Predictions vs. Ground Truth**

<p align="center">
  <em>The model demonstrates high accuracy in segmenting cancerous regions, closely matching the ground truth masks provided by radiologists.</em>
</p>
<p align="center">
  <img src="results/prediction_examples.png" width="800" />
</p>

---

## 1. Project Overview

This project implements a state-of-the-art **DSU-Net (Dense Skip U-Net with Attention)** for the automated segmentation of Regions of Interest (ROI) in breast cancer mammograms. Using the large-scale public **CBIS-DDSM dataset**, the model is trained to accurately identify and outline mass and calcification areas, serving as a powerful proof-of-concept for a computer-aided diagnosis (CAD) tool.

The entire workflow, from exploratory data analysis to final model evaluation, is documented in the accompanying Jupyter Notebook, showcasing a professional, reproducible, and robust approach to solving a real-world medical imaging problem.

---

## 2. Key Features

-   **🔬 Advanced Model Architecture:** A custom DSU-Net is implemented from scratch in PyTorch, incorporating **Dense Blocks** for improved feature propagation and **Attention Gates** to help the model focus on the most salient features.
-   **💧 Leak-Proof Data Splitting:** A rigorous **patient-aware** validation split is used to ensure the model generalizes to unseen patients, preventing the common pitfall of data leakage and providing a reliable performance metric.
-   **⚖️ Sophisticated Loss Function:** A custom-weighted **Combined Loss** function (0.5 * Dice Loss + 0.3 * Focal Loss + 0.2 * BCE) is implemented to handle both severe pixel-level imbalance and mild case-level imbalance.
-   **⚙️ State-of-the-Art Training Regimen:** The training loop employs modern best practices, including a `ReduceLROnPlateau` learning rate scheduler, gradient clipping for stability, and an **Early Stopping** mechanism to prevent overfitting.
-   **🖼️ Rich Data Augmentation:** A comprehensive set of augmentations from the `albumentations` library is used to create a more robust model that generalizes well to variations in imaging data.

---

## 3. Results & Visualizations

The model was trained on the Kaggle platform, achieving excellent performance on the held-out validation set.

#### **Final Metrics**

| Metric                       | Best Score Achieved |
| :--------------------------- | :------------------ |
| **Validation Dice Score**    | **0.9428**          |
| **Validation IoU Score**     | **0.9114**          |

#### **Training Performance**
The learning curves demonstrate a healthy training process with no signs of significant overfitting, a result of the robust validation strategy and data augmentation.

![Training Curves](results/training_curves.png)

---

## 4. Tech Stack

-   **Python 3.9+**
-   **PyTorch:** For building and training the deep learning model.
-   **pandas & NumPy:** For data manipulation and numerical operations.
-   **OpenCV:** For image processing.
-   **Albumentations:** For the data augmentation pipeline.
-   **Matplotlib & Seaborn:** For data visualization.
-   **scikit-learn:** For data splitting.
-   **Kaggle:** For the cloud-based GPU-enabled training environment.

---

## 5. About the Dataset

This project utilizes the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** dataset, a large, standardized collection of mammography images for breast cancer research.

-   **Source:** [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
-   **Kaggle Mirror:** [CBIS-DDSM Breast Cancer Image Dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

**Note:** Due to its significant size (~150 GB), the dataset is not included in this repository. Please follow the instructions below to run the code in an environment with the data.

---

## 6. How to Run This Project

### Method 1: Run on Kaggle (Recommended)

The simplest way to run this project is to use the same Kaggle environment where it was developed.

1.  **Download the Notebook:** Download the `DSU_Net_Segmentation_Workflow.ipynb` file from the `notebooks/` directory in this repository.
2.  **Navigate to Kaggle:** Go to the [CBIS-DDSM dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset).
3.  **Create and Upload:** Click `New Notebook`, then in the new environment, go to `File > Upload Notebook` and select the `.ipynb` file you downloaded.
4.  The notebook is now ready to run in an environment with the data and necessary GPU resources already attached.

### Method 2: Run Locally (Advanced)

Running this project locally requires a significant amount of disk space and a CUDA/MPS-enabled GPU.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AIchemizt/Attention-U-Net-for-ROI-Analysis.git
    cd Attention-U-Net-for-ROI-Analysis
    ```

2.  **Set Up Environment:**
    ```bash
    # It is highly recommended to use a virtual environment
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # pydensecrf requires a special installation
    pip install git+https://github.com/lucasb-eyer/pydensecrf.git
    ```

4.  **Download the Data:**
    -   Download the full JPEG version of the dataset from the [Kaggle link](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) above.
    -   Create a `data/` directory in the project root.
    -   Unzip the dataset and ensure the final structure is `data/csv/` and `data/jpeg/`.

5.  **Run the Notebook:**
    -   Launch Jupyter and open the `DSU_Net_Segmentation_Workflow.ipynb` notebook from the `notebooks/` directory.

---

## 7. Directory Structure
Attention-U-Net-for-ROI-Analysis/
│
├── DSU_Net_Segmentation_Workflow.ipynb (Main project notebook)
│
├── prediction_examples.png (Saved image of model predictions)
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt


---

## 8. License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 9. Contact

Abhishek Chandel – [LinkedIn](https://www.linkedin.com/in/abhishek-chandel-0b0a63127/)