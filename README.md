 # AI-Driven-Web-Application-for-Automated-Disease-Detection-in-Rice-and-Pulse-Crops_Nov_Batch-6_2025
 
 🌾 **AI-Driven Crop Disease Detection (Rice & Pulses)**

### *Milestone-1 – Model Training & Evaluation*

**Intern:** Sai Venu Gopala Swamy
**Organization:** Infosys – AI Internship Program

---

## 📌 **Project Overview**

This project aims to develop an **AI-powered disease detection system** for:

* **Rice crops**
* **Pulse crops (BPLD + Pea Plant)**

The goal is to automate leaf disease diagnosis using **Deep Learning models** trained on curated datasets of plant leaf images.

This repository contains all deliverables for **Milestone-1**, including:

* Dataset preparation
* Train/val/test dataset splitting
* Model training for Rice and Pulses
* Accuracy results
* Saved model weights (.pth files)

---

## 📁 **Repository Structure**

```
AI_Crop_Disease/
│
├── dataset_split/                 # Final cleaned & reduced dataset used for training
│   ├── RICE/
│   └── PULSES/
│
├── 2 split codes/                 # Scripts for dataset split & reduction
│   ├── split_dataset.py
│   ├── reduce_rice_dataset.py
│   ├── reduce_pulses_dataset.py
│   └── dataset_loader.py
│
├── train_rice_model.py            # Final Rice model training script
├── train_pulses_model.py          # Final Pulses model training script
│
├── rice_model.pth                 # Saved Rice model (best validation accuracy)
├── pulses_model.pth               # Saved Pulses model
│
├── rice_training_results.txt      # Rice accuracy report
├── pulses_training_results.txt    # Pulses accuracy report
│
└── README.md                      # Project documentation
```

---

## 🧠 **Model Architecture**

Two separate CNN models were trained:

### **1️⃣ Rice Model**

* Custom Improved CNN
* Strong augmentation
* Lightweight architecture (CPU-friendly)
* Trained on reduced dataset (30–40 images/class)

### **2️⃣ Pulses Model**

* Custom CNN optimized for small datasets
* Handles multiple leaf disease classes from BPLD + Pea Plant
* Heavy augmentation to improve generalization

---

## 📊 **Training & Validation Results**

Both models save their results into text files:

### 📄 Rice:

```
rice_training_results.txt
```

### 📄 Pulses:

```
pulses_training_results.txt
```

These contain:

* Best Validation Accuracy
* Final Training Accuracy
* Model file saved

---

## ▶️ **How to Run Training Scripts**

### **Train Rice Model**

```
py -3.10 train_rice_model.py
```

### **Train Pulses Model**

```
py -3.10 train_pulses_model.py
```

Both scripts automatically:

* Load dataset
* Train for 25 epochs
* Save best .pth model
* Generate .txt summary of results

---

## 🧪 **Evaluation**

A separate evaluation script (optional for Milestone-1) can compute:

* Test accuracy
* Per-class metrics
* Confusion matrix

*(If needed, GPT can generate this script.)*

---

## 🚀 **Milestone-1 Deliverables Completed**

| Task                                           | Status                |
| ---------------------------------------------- | --------------------- |
| Dataset cleaning and reduction                 | ✅ Done                |
| Train/Val/Test split                           | ✅ Done                |
| Rice model training                            | ✅ Completed           |
| Pulses model training                          | ✅ Completed           |
| Accuracy reports (.txt)                        | ✅ Generated           |
| Model files (.pth)                             | ✅ Saved               |
| Code uploaded to GitHub (branch: **sai-venu**) | ⏳ Pending (next step) |
| Results uploaded to Google Drive               | ⏳ Pending             |

---

## 🧾 **Branch Information**

As instructed by the mentor:

```
Branch Name: sai-venu
```

All Milestone-1 files must be pushed to this branch.

---

## ✨ **Future Enhancements (Milestone-2 & 3)**

* Train combined universal model
* Improve accuracy using MobileNetV2 or EfficientNet
* Build a Streamlit web-app
* Deploy final AI model

---

## 🙏 Acknowledgements

This work is developed as part of the **Infosys Internship Program**, under the guidance of the AI project mentors.


