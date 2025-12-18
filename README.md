  **AI-Driven Crop Disease Detection (Rice & Pulses)**

### *Milestone-1 – Model Training & Evaluation*

**Intern:** Sai Venu Gopala Swamy
**Organization:** Infosys – AI Internship Program

---

##  **Project Overview**

This project aims to develop an **AI-powered disease detection system** for:

* **Rice crops**
* **Pulse crops**

The goal is to automate leaf disease diagnosis using **Deep Learning models** trained on curated datasets of plant leaf images.

This repository contains all deliverables for **Milestone-1**, including:

* Dataset preparation
* Train/val/test dataset splitting
* Model training for Rice and Pulses
* Accuracy results
* Saved model weights (.pth files)

---

##  **Repository Structure**

```


```
AI-Driven-Web-Application-for-Automated-Disease-Detection-in-Rice-and-Pulse-Crops/
│
├── dataset_split/                     # Final cleaned & reduced dataset used for training
│   ├── RICE/
│   └── PULSES/
│
├── split_codes/                       # Scripts for dataset splitting & reduction
│   ├── split_dataset.py
│   ├── reduce_rice_dataset.py
│   ├── reduce_pulses_dataset.py
│   └── dataset_loader.py
│
├── training_scripts/                  # Model training scripts
│   ├── train_rice_model.py
│   └── train_pulses_model.py
│
├── results/                           # Training & testing result reports
│   ├── rice_training_results.txt
│   └── pulses_training_results.txt
│
├── streamlit_app/                     # Streamlit web application
│   ├── app.py                         # Streamlit UI and navigation
│   ├── auth.py                        # Authentication and SQLite DB handling
│   ├── model_predict.py               # ML prediction logic (SOLID + OOPS)
│   ├── requirements.txt               # Python dependencies
│   ├── models/                        # Trained ML models
│   │   ├── rice_model_improved.pth
│   │   └── pulses_model_improved.pth
│   └── db/
│       └── users.db                   # SQLite database
│
└── README.md                          # Project documentation


```
##  SOLID Principles Implementation in This Project

This project follows **SOLID principles** using **Object-Oriented Programming (OOPS)** concepts to ensure scalability, maintainability, and clean architecture.

---

### 1️ Single Responsibility Principle (SRP)

> *Each module has only one responsibility.*

**Implementation in the project:**

* `app.py`

  * Handles **Streamlit UI**, navigation, and user interaction only.
* `auth.py`

  * Handles **user authentication**, login, registration, and SQLite database operations.
* `model_predict.py`

  * Handles **model loading and prediction logic** only.
* `training_scripts/`

  * Contains scripts used **only for training models**.
* `dataset_split/`

  * Contains datasets used **only for training and evaluation**.

 This separation ensures that changes in one module do not affect others.

---

### 2️ Open / Closed Principle (OCP)

> *Software entities should be open for extension but closed for modification.*

**Implementation in the project:**

* A **base abstract class** is used for prediction logic.
* New crop disease models (e.g., Wheat, Tomato) can be added by **creating new subclasses**.
* Existing UI and prediction flow do **not need to be modified**.

 This allows easy extension of the system without rewriting existing code.

---

### 3️ Liskov Substitution Principle (LSP)

> *Derived classes must be substitutable for their base classes.*

**Implementation in the project:**

* `RiceDiseaseModel` and `PulsesDiseaseModel` inherit from a common abstract base class.
* Both models implement the same `predict()` interface.
* Either model can be used interchangeably without breaking the application.

 This ensures consistent behavior across different crop models.

---

### 4️ Interface Segregation Principle (ISP)

> *Clients should not be forced to depend on interfaces they do not use.*

**Implementation in the project:**

* The Streamlit UI (`app.py`) interacts with:

  * `verify_user()` and `add_user()` from `auth.py`
  * `predict()` from `model_predict.py`
* UI does **not depend on**:

  * Database query logic
  * PyTorch model internals
  * CNN architecture details

 Each module exposes only the necessary interfaces.

---

### 5️ Dependency Inversion Principle (DIP)

> *High-level modules should not depend on low-level modules; both should depend on abstractions.*

**Implementation in the project:**

* The Streamlit UI depends on an **abstract prediction interface**, not on specific CNN implementations.
* Prediction logic depends on abstract base classes rather than concrete model details.
* This reduces tight coupling between UI, ML models, and database layers.

    This makes the system flexible and easier to maintain.

---

##  Use of OOPS Concepts

The project applies the following **Object-Oriented Programming concepts**:

* **Abstraction**

  * Common prediction behavior is defined using an abstract base class.
* **Inheritance**

  * Crop-specific models inherit from the base prediction class.
* **Polymorphism**

  * Different crop models implement the same prediction interface.
* **Encapsulation**

  * Internal model logic is hidden from the UI layer.

---


