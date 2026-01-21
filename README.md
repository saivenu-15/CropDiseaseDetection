
---

# **AI-Driven Crop Disease Detection (Rice & Pulses)**

**Intern:** Sai Venu Gopala Swamy
**Organization:** Infosys – AI Internship Program

---

##  Project Overview
 Project Overview

This project is an AI-powered web application designed to detect crop diseases in rice and pulse crops using leaf images.
It leverages deep learning (CNNs) for accurate disease classification and provides confidence scores, disease insights, treatment suggestions, and downloadable reports through an interactive web interface.

The system aims to assist farmers, agricultural experts, and researchers in early disease detection to reduce crop losses and improve productivity.

The project is designed following **SOLID principles** and **Object-Oriented Programming (OOPS)** concepts to ensure clean architecture, scalability, and maintainability.

---

##  Objectives

* Automate crop disease identification using AI
* Reduce manual inspection efforts
* Provide a simple and secure web interface for predictions
* Build a scalable system following software engineering best practices

---
.

🎯 Key Features

📷 Image-based crop disease detection

🌱 Supports Rice & Pulse crops

🤖 CNN-based deep learning model

📊 Confidence scoring for predictions

🧪 Disease causes, symptoms, prevention & treatment advice

📄 Downloadable disease report (PDF)

🔐 User authentication (login & signup)

💬 Integrated chatbot for guidance

🌐 Interactive Streamlit-based UI
---

##  Repository Structure

```text

AI-Driven-Web-Application-for-Automated-Disease-Detection
│
├── db/                     # Database files (user authentication)
│
├── models/                 # Trained deep learning model files (.pth)
│
├── results/                # Model training and evaluation results
│
├── streamlit_app/          # Streamlit-specific assets (if any)
│
├── test_scripts/           # Model testing scripts
│
├── training_scripts/       # Model training scripts
│
├── app.py                  # Main Streamlit application entry point
├── auth.py                 # User authentication logic
├── chatbot.py              # AI chatbot module
├── disease_advice.py       # Disease recommendations & guidance
├── model_predict.py        # Model loading & prediction logic
├── pdf_report.py           # PDF report generation
│
├── Finalized_PPT_Final.pdf # Final project presentation
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation


---

🚀 How to Run the Project

1️⃣ Clone the Repository
git clone <repository-url>
cd AI-Driven-Web-Application-for-Automated-Disease-Detection

2️⃣ Install Dependencies
pip install -r requirements.txt
pip install -r streamlit_app/requirements.txt

3️⃣ Run the Application

streamlit run app.py

##  Model Training Summary

* Separate CNN models trained for:

  * Rice disease classification
  * Pulses disease classification
* Datasets were **cleaned, reduced, and split** into train/validation/test sets
* Best-performing models were saved using `.pth` format
* Accuracy results are documented in the `results/` directory

---

##  Streamlit Web Application Features

*  User Authentication (Login / Signup)
*  Image Upload (JPG / PNG)
*  AI-based Disease Prediction
*  Clear Prediction Output
*  Logout Functionality
*  SQLite Database for credential storage

---

##  SOLID Principles Implementation

This project strictly follows **SOLID principles** using **OOPS concepts** for clean and maintainable design.

---

### 1️ Single Responsibility Principle (SRP)

> *Each module has one and only one responsibility.*

**Implementation:**

* `app.py` → UI, navigation, user interaction
* `auth.py` → Authentication and database handling
* `model_predict.py` → Model loading and prediction logic
* `training_scripts/` → Model training only
* `dataset_split/` → Dataset storage only

✔ Changes in one module do not affect others.

---

### 2️ Open / Closed Principle (OCP)

> *Open for extension, closed for modification.*

**Implementation:**

* Prediction logic is designed so that **new crop models** can be added easily
* Existing code does not need modification to support future crops (e.g., wheat, tomato)

✔ System is extensible without breaking existing functionality.

---

### 3️ Liskov Substitution Principle (LSP)

> *Derived classes must be substitutable for base classes.*

**Implementation:**

* Rice and Pulses models follow the same prediction interface
* Either model can be used interchangeably by the UI

✔ Ensures consistent behavior across models.

---

### 4️ Interface Segregation Principle (ISP)

> *Clients should not depend on unnecessary interfaces.*

**Implementation:**

* UI interacts only with:

  * `verify_user()` / `add_user()` from `auth.py`
  * `predict()` from `model_predict.py`
* UI does **not depend on**:

  * Database internals
  * CNN architecture details

✔ Loose coupling between modules.

---

### 5️ Dependency Inversion Principle (DIP)

> *High-level modules depend on abstractions, not implementations.*

**Implementation:**

* Streamlit UI depends on an abstract prediction interface
* Prediction logic abstracts model details from UI

✔ Makes the system flexible and easy to maintain.

---

##  Use of OOPS Concepts

The project applies the following OOPS concepts:

* **Abstraction** – Common prediction behavior defined via base logic
* **Inheritance** – Crop-specific models extend shared prediction behavior
* **Polymorphism** – Same prediction interface used for different crops
* **Encapsulation** – Internal model logic hidden from UI

---

##  How to Run the Application

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

##  Conclusion

This project demonstrates the practical application of:

* Deep Learning
* Streamlit web development
* SOLID principles
* Object-Oriented Programming
* Clean software architecture

It provides a scalable foundation for real-world AI-based agricultural solutions.

---




