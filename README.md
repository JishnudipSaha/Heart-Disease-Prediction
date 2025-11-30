#  **Heart Disease Prediction using Machine Learning**

A complete end-to-end Machine Learning project to predict the presence of heart disease using clinical patient data.
Multiple ML models were trained, tuned, compared, and evaluated to identify the most accurate and reliable classifier.

The **Neural Network (MLPClassifier)** emerged as the best-performing model.

---

#  **Table of Contents**

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Tech Stack](#tech-stack)
4. [Project Workflow](#project-workflow)
5. [Model Training](#model-training)
6. [Hyperparameter Tuning Results](#hyperparameter-tuning-results)
7. [Model Performance Comparison](#model-performance-comparison)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [Contact](#contact)

---

#  **Project Overview**

Heart disease is one of the leading causes of death globally.
Early diagnosis can drastically improve treatment outcomes.

This project uses machine learning methods to classify whether a patient is at high risk of heart disease based on multiple medical attributes. The pipeline includes:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Preprocessing
* Model building
* Hyperparameter tuning
* Model comparison

---

#  **Dataset Description**

The dataset includes clinical parameters such as:

| Feature    | Description                 |
| ---------- | --------------------------- |
| age        | Patient age                 |
| sex        | Gender                      |
| cp         | Chest pain type             |
| trestbps   | Resting blood pressure      |
| chol       | Cholesterol                 |
| fbs        | Fasting blood sugar         |
| restecg    | Rest ECG results            |
| thalach    | Maximum heart rate          |
| exang      | Exercise induced angina     |
| oldpeak    | ST depression               |
| slope      | Slope of peak exercise      |
| ca         | Major vessels count         |
| thal       | Thalassemia                 |
| **target** | 1 = Disease, 0 = No disease |

No missing values were present; duplicates were removed.

---

#  **Tech Stack**

* **Python**
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-Learn
* ydata-profiling (Automated EDA)
* MLPClassifier
* GridSearchCV

---

#  **Project Workflow**

### ✔ 1. Load Dataset

### ✔ 2. Clean & Remove Duplicates

### ✔ 3. Perform Automated EDA

### ✔ 4. Feature/Target Separation

### ✔ 5. Train-Test Split

### ✔ 6. Train ML Models

### ✔ 7. Perform Hyperparameter Tuning

### ✔ 8. Compare Metrics

### ✔ 9. Select Best Model

---

#  **Model Training**

The following models were trained inside **Scikit-Learn Pipelines**:

* Logistic Regression
* Decision Tree
* Random Forest
* Neural Network (MLPClassifier)

All models were standardized using **StandardScaler**.

---

# 🔧 **Hyperparameter Tuning Results**

Below are the exact tuning outputs from your notebook.

---

## ## 1️⃣ Logistic Regression

### **Best Parameters**

```json
{
  "model__C": 0.1,
  "model__penalty": "l2",
  "model__solver": "lbfgs"
}
```

### **CV Result**

* **Best CV F1-score:** 0.86296

### **Test Performance**

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.7869 |
| Precision | 0.7500 |
| Recall    | 0.9091 |
| F1-Score  | 0.8219 |
| ROC-AUC   | 0.8788 |

---

## 2️⃣ Decision Tree

### **Best Parameters**

```json
{
  "model__criterion": "entropy",
  "model__max_depth": 3,
  "model__min_samples_leaf": 1,
  "model__min_samples_split": 2
}
```

### **CV Result**

* **Best CV F1-score:** 0.81699

### **Test Performance**

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.7869 |
| Precision | 0.7631 |
| Recall    | 0.8788 |
| F1-Score  | 0.8169 |
| ROC-AUC   | 0.8561 |

---

## 3️⃣ Random Forest

### **Best Parameters**

```json
{
  "model__max_depth": 5,
  "model__min_samples_leaf": 4,
  "model__min_samples_split": 2,
  "model__n_estimators": 300
}
```

### **CV Result**

* **Best CV F1-score:** 0.86856

### **Test Performance**

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.8197 |
| Precision | 0.7619 |
| Recall    | 0.9697 |
| F1-Score  | 0.8533 |
| ROC-AUC   | 0.9015 |

---

## 4️⃣ Neural Network (MLPClassifier)

### **Best Parameters**

```json
{
  "model__activation": "tanh",
  "model__alpha": 0.0001,
  "model__hidden_layer_sizes": [16],
  "model__learning_rate_init": 0.001
}
```

### **CV Result**

* **Best CV F1-score:** 0.83934

### **Test Performance**

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.8525 |
| Precision | 0.8158 |
| Recall    | 0.9394 |
| F1-Score  | 0.8732 |
| ROC-AUC   | 0.8723 |

---

# 📊 **Model Performance Comparison**

### **Overall Metric Comparison**

| Model                    | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| ------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression      | 0.7869     | 0.7500     | 0.9091     | 0.8219     | 0.8788     |
| Decision Tree            | 0.7869     | 0.7631     | 0.8788     | 0.8169     | 0.8561     |
| Random Forest            | 0.8197     | 0.7619     | **0.9697** | 0.8533     | **0.9015** |
| **Neural Network (MLP)** | **0.8525** | **0.8158** | 0.9394     | **0.8732** | 0.8723     |

---

#  **Best Model: Neural Network (MLPClassifier)**

Reasons:

* Highest **F1-score**
* Best real-world balance of **Precision vs Recall**
* Strong generalization
* Handles non-linear patterns well

---

#  **Conclusion**

This project demonstrates a complete Machine Learning pipeline for Heart Disease Prediction.
After evaluating multiple classifiers, the **MLP Neural Network** emerged as the best-performing model.

The project proves how ML can assist healthcare professionals with data-driven diagnosis support.

---

# 🔮 **Future Work**

* Include more clinical features
* Use XGBoost / LightGBM for deeper analysis
* Deploy as a **Streamlit / Flask** web app
* Add explainability using **SHAP**
* Train on larger multi-hospital datasets

---

# 📬 **Contact**

If you'd like help deploying or enhancing this project:

**Developer:** Jishnudip
Feel free to connect anytime 🙌
