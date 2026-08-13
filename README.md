# **Student Dropout Prediction Using Machine Learning**

## **1. Introduction**

Student dropout is an important challenge faced by educational institutions. Identifying students who are at risk of dropping out can help institutions provide timely academic support and improve student retention.

This project develops a **machine learning-based student outcome prediction system** that analyzes student academic, demographic, and socioeconomic information to predict one of three outcomes:

* **Dropout**
* **Enrolled**
* **Graduate**

The project uses **XGBoost** as the machine learning model, **SHAP** for explainability, and **Streamlit** to provide an interactive user interface.

---

## **2. Problem Statement**

Educational institutions collect large amounts of student information, but manually analyzing this data to identify students who may be at risk is difficult and time-consuming.

The problem addressed by this project is:

> **To develop a machine learning system capable of predicting student academic outcomes and providing understandable insights into the factors influencing those predictions.**

The system aims to support early identification and data-driven academic decision-making.

---

## **3. Methodology**

The project follows a supervised machine learning methodology.

### **Dataset Preparation**

The dataset contains **4,424 student records and 37 columns**, including academic, demographic, enrollment, financial, and socioeconomic information.

The dataset is checked for missing values and duplicate records before model development.

### **Data Preprocessing**

The dataset is divided into:

* **Input features (X)**
* **Target variable (y)**

The target classes are encoded as:

```text
0 → Dropout
1 → Enrolled
2 → Graduate
```

The data is then divided into training and testing sets using an **80:20 stratified split**.

### **Model Development**

**XGBoost Classifier** is used to train the multi-class classification model.

The trained model learns patterns from historical student data and predicts the likely academic outcome of new student profiles.

### **Explainable AI**

**SHAP (SHapley Additive exPlanations)** is used to interpret model predictions and identify the features that contribute to the predicted outcome.

### **Application Development**

A **Streamlit application** provides an interactive interface where users can enter student information and obtain predictions and explanations.

### **Workflow**

```text
Student Dataset
      ↓
Data Preprocessing
      ↓
Feature & Target Separation
      ↓
Target Encoding
      ↓
Train/Test Split
      ↓
XGBoost Model
      ↓
Prediction
      ↓
SHAP Explanation
      ↓
Streamlit Application
```

---

## **4. Dependencies**

The project is developed using Python and the following libraries:

| Dependency       | Purpose                            |
| ---------------- | ---------------------------------- |
| **Python**       | Core programming language          |
| **Pandas**       | Data processing                    |
| **NumPy**        | Numerical operations               |
| **Scikit-learn** | Preprocessing and model evaluation |
| **XGBoost**      | Machine learning classification    |
| **SHAP**         | Model explainability               |
| **Streamlit**    | Interactive web application        |
| **Matplotlib**   | Data visualization                 |
| **OpenPyXL**     | Excel dataset handling             |

Install the dependencies using:

```bash
pip install pandas numpy scikit-learn xgboost shap streamlit matplotlib openpyxl
```

Run the application using:

```bash
streamlit run app.py
```

---

## **5. Deliverables**

The project delivers:

* **Machine Learning Model** — XGBoost-based student outcome classifier.
* **Prediction System** — Predicts Dropout, Enrolled, or Graduate.
* **Explainable AI Module** — SHAP-based prediction explanations.
* **Interactive Web Application** — Streamlit-based user interface.
* **Student Profile Analysis** — Allows student information to be entered and analyzed.
* **Prediction Probabilities** — Displays the probability associated with each outcome.
* **Student Comparison** — Supports comparison of different student profiles.
* **Project Documentation** — Complete GitHub documentation and dataset appendix.

---

## **6. Importance of the Project**

The project demonstrates how machine learning can be applied to educational data to support early identification of potential student outcomes.

### **For Educational Institutions**

Helps identify patterns associated with student dropout and retention.

### **For Academic Advisors**

Provides additional information that can support student evaluation and intervention.

### **For Students**

Potentially enables earlier academic support for students who may require additional assistance.

### **For Data Science**

Demonstrates a complete machine learning workflow from data preprocessing to deployment and explainability.

### **For Explainable AI**

Shows how a machine learning model can provide not only a prediction but also information about the factors influencing that prediction.

> The system is intended as a decision-support tool and should not replace human academic judgment.

---

# **7. Appendix**

## **Appendix A — Dataset Summary**

| Property       | Value |
| -------------- | ----: |
| Total Records  | 4,424 |
| Total Columns  |    37 |
| Input Features |    36 |
| Target Classes |     3 |
| Missing Values |     0 |
| Duplicate Rows |     0 |
| Training Data  |   80% |
| Testing Data   |   20% |
| Random State   |    42 |

---

## **Appendix B — Target Distribution**

| Target    | Number of Students | Percentage |
| --------- | -----------------: | ---------: |
| Graduate  |              2,209 |     49.93% |
| Dropout   |              1,421 |     32.12% |
| Enrolled  |                794 |     17.95% |
| **Total** |          **4,424** |   **100%** |

---

## **Appendix C — Target Encoding**

| Encoded Value | Academic Outcome |
| ------------: | ---------------- |
|         **0** | Dropout          |
|         **1** | Enrolled         |
|         **2** | Graduate         |

---

## **Appendix D — Dataset Features**

|  # | Feature                               | Description                       |
| -: | ------------------------------------- | --------------------------------- |
|  1 | Marital Status                        | Student's marital status          |
|  2 | Application Mode                      | Method used for application       |
|  3 | Application Order                     | Order of application              |
|  4 | Course                                | Selected academic course          |
|  5 | Daytime/Evening Attendance            | Attendance schedule               |
|  6 | Previous Qualification                | Previous academic qualification   |
|  7 | Previous Qualification Grade          | Grade from previous qualification |
|  8 | Nationality                           | Student nationality               |
|  9 | Mother's Qualification                | Mother's qualification            |
| 10 | Father's Qualification                | Father's qualification            |
| 11 | Mother's Occupation                   | Mother's occupation               |
| 12 | Father's Occupation                   | Father's occupation               |
| 13 | Admission Grade                       | Admission grade                   |
| 14 | Displaced                             | Relocation indicator              |
| 15 | Educational Special Needs             | Special needs indicator           |
| 16 | Debtor                                | Financial debt indicator          |
| 17 | Tuition Fees Up to Date               | Tuition payment status            |
| 18 | Gender                                | Student gender                    |
| 19 | Scholarship Holder                    | Scholarship status                |
| 20 | Age at Enrollment                     | Age when enrolled                 |
| 21 | International                         | International student indicator   |
| 22 | 1st Semester Units Enrolled           | First semester enrollment         |
| 23 | 1st Semester Units Evaluated          | First semester evaluations        |
| 24 | 1st Semester Units Approved           | First semester approved units     |
| 25 | 1st Semester Grade                    | First semester grade              |
| 26 | 1st Semester Units Without Evaluation | Units without evaluation          |
| 27 | 2nd Semester Units Enrolled           | Second semester enrollment        |
| 28 | 2nd Semester Units Evaluated          | Second semester evaluations       |
| 29 | 2nd Semester Units Approved           | Second semester approved units    |
| 30 | 2nd Semester Grade                    | Second semester grade             |
| 31 | 2nd Semester Units Without Evaluation | Units without evaluation          |
| 32 | Unemployment Rate                     | Unemployment indicator            |
| 33 | Inflation Rate                        | Inflation indicator               |
| 34 | GDP                                   | Gross Domestic Product indicator  |
| 35 | Target                                | Final academic outcome            |

> **Note:** The feature descriptions should be aligned with the exact column names present in the project dataset.

---

## **Appendix E — Technology Stack**

| Technology   | Role                       |
| ------------ | -------------------------- |
| Python       | Programming                |
| Pandas       | Data processing            |
| NumPy        | Numerical computation      |
| Scikit-learn | Machine learning utilities |
| XGBoost      | Classification model       |
| SHAP         | Explainable AI             |
| Streamlit    | Web interface              |
| Matplotlib   | Visualization              |
| Git & GitHub | Version control            |

---

## **Appendix F — Project Structure**

```text
student-dropout-prediction/
│
├── app.py
├── edu.py
├── data.csv.xlsx
├── .gitignore
└── README.md
```

---

## **Author**

**Sanjay P**
B.Tech — Artificial Intelligence and Data Science

**GitHub:** Sanjay-4510

---

**Predict • Explain • Understand**
