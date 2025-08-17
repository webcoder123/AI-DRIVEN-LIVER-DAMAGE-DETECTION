# 🩺 Liver Damage Detection using Deep Learning

## 📌 Project Overview
Traditional liver damage detection relies on **manual analysis of histopathology images**, which is time-consuming and prone to error.  
This project leverages **Deep Learning (CNNs)** to automate liver damage detection, assisting hepatologists in **early diagnosis** and reducing manual workload.

The model classifies liver images into:
- **Normal Liver**
- **Hepatocellular Carcinoma (HCC)**
- **Cholangiocarcinoma (CC)**

---

## 🎯 Objectives
- **Maximize early detection accuracy** using AI-driven models.  
- **Minimize false positives/negatives** for reliable diagnostics.  
- **Reduce manual diagnostic efforts by 20–30%** through automation.  
- **Achieve >85% prediction accuracy** (target exceeded with 97%).  

---

## 🏗️ Project Workflow
1. Image Collection  
2. Exploratory Data Analysis (EDA)  
3. Data Preprocessing & Augmentation  
4. Model Building (CNNs & Transfer Learning)  
5. Model Evaluation  
6. Deployment with Streamlit  
7. Monitoring & Maintenance  

---

## 🖼️ Dataset
- **Source**: Manually collected histopathology images (not available on Kaggle).  
- **Original Size**: 270 images (161 MB).  
  - HCC: 79  
  - CC: 170  
  - Normal: 21  

### Augmentation & Balancing
- Applied **10 transformations per image** (blur, shift, crop, padding, normalization, rotation, flips, transpose).  
- Expanded dataset → **6,006 images** (~1.6 GB).  
- Balanced classes: **1003 images per class** (undersampling).  

### Final Split
- Train: 70% (4,203 images)  
- Test: 20% (1,201 images)  
- Validation: 10% (602 images)  

---

## 🔧 Data Preprocessing
- Resizing & normalization of images.  
- Data augmentation (rotation, flips, brightness adjustment).  
- Balancing strategies to avoid class bias.  

---

## 🤖 Models Used
Pre-trained CNN models (ImageNet) fine-tuned for classification:
- 🧠 **DenseNet121** → ✅ Best model (97% accuracy)  
- 🧠 ResNet50  
- 🧠 ResNet101  
- 🧠 ResNet152  
- 🧱 VGG16  

### Model Performance (Final)
| Model       | Train Acc. | Val Acc. | Test Acc. |
|-------------|------------|----------|-----------|
| DenseNet121 | **97.0%**  | **96.8%**| **97.0%** |
| VGG16       | 70.1%      | 68.9%    | 69.6%     |
| ResNet101   | 63.8%      | 61.4%    | 59.3%     |
| ResNet50    | 61.2%      | 59.1%    | 57.3%     |
| ResNet152   | 57.1%      | 54.6%    | 55.4%     |

---

## 🚀 Deployment
- **Framework**: Streamlit  
- **Features**:  
  - Upload histopathology images (JPG/PNG).  
  - Real-time classification (Normal, HCC, CC).  
  - Display prediction confidence.  

To run locally:
```bash

# Run Streamlit app
streamlit run app.py
📊 Challenges
Limited Image Availability → Manually curated dataset.

Data Imbalance → Augmentation & undersampling strategies.

Image Quality Variations → Preprocessing (resizing, normalization).

Computational Cost → Required GPU acceleration.

Fine-tuning Complexity → Careful hyperparameter optimization.

✅ Conclusion
Achieved 97% accuracy with DenseNet121.

Robust and generalizable performance on unseen test images.

Real-time, Streamlit-based deployment for clinical usage.

Demonstrates potential for integration into healthcare diagnostics.
