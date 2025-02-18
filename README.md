# Multi-Model Image Classification using Pre-Trained Feature Extractors

## Project Overview
This project focuses on **multi-class image classification** using **feature vectors extracted from pre-trained models**. Utilize embeddings from **CLIP, DINOv2, ResNet, and ViT** to classify **40,000 images into 10 categories**.

To evaluate different approaches, implemented and compared three models:
- **XGBoost**
- **LightGBM**
- **Multi-Layer Perceptron (MLP)**

**Optuna** was used for **hyperparameter tuning**, and models were evaluated using **5-fold cross-validation**. The **MLP classifier** achieved the highest **macro F1 score of 0.9900**, making it the best-performing model.


## Dataset
The dataset consists of **40,000 images**, each represented by feature vectors extracted using four pre-trained models. A **20,000-image validation-test set** was also provided.

### **Feature Extractors Used:**
1. **CLIP** – Captures **semantic and visual** information.
2. **DINOv2** – Provides **structural and object-level patterns**.
3. **ResNet** – Extracts **hierarchical and detailed patterns**.
4. **ViT** – Processes images **through smaller patches**, capturing **spatial relationships**.

### **Feature Vectors Dimensions:**
- **ResNet:** (40,000, 512)  
- **ViT:** (40,000, 768)  
- **CLIP:** (40,000, 512)  
- **DINO:** (40,000, 768)  

*All feature vectors were concatenated into a tabular format for training models.*

---

## Methodology
### Data Preprocessing
- Feature vectors were **stacked into a unified matrix** for training and testing.
- No **dimensionality reduction** was applied to retain full feature space.
- No **image augmentation** was applied, as the focus was on extracted embeddings.

### Models Used
We tested three models to classify images based on the extracted feature vectors.

#### **1. XGBoost (Extreme Gradient Boosting)**
- Handles tabular data well.
- Trained using **5-fold cross-validation**.
- **Macro F1 Score:** **0.9814**

#### **2. LightGBM (Gradient Boosting Model by Microsoft)**
- Faster than XGBoost, optimized for large-scale data.
- **Macro F1 Score:** **0.9842**

#### **3. Multi-Layer Perceptron (MLP) (Best Model)**
- **Deep learning model** optimized using **Optuna**.
- Used **batch normalization** and **dropout** to prevent overfitting.
- **Macro F1 Score:** **0.9900**

### Hyperparameter Tuning (Optuna)
For **MLP**, optimized:

✅ **Number of hidden layers**  
✅ **Learning rate**  
✅ **Batch size**  

*Final MLP model achieved the best performance with the highest stability across folds.*

---

## Results & Performance
The models were evaluated using **macro F1 scores** across 5-fold cross-validation.

| **Model**   | **Macro F1 Score**  |
|------------|-------------------|
| **MLP**     | **0.9900**         |
| **LightGBM** | **0.9842**         |
| **XGBoost**  | **0.9814**         |

*The narrow spread of F1 scores for MLP indicates **low variance** and high stability, making it the best-performing model.*
