# 🤖 GlucoAI – Smart Diabetes Detection System

GlucoAI is an intelligent diagnostic tool that predicts the likelihood of diabetes in individuals using machine learning. It processes clinical parameters such as glucose levels, BMI, insulin, age, and blood pressure to deliver accurate, real-time predictions with explainable results. Built using Python, Scikit-learn, SHAP, and Streamlit, GlucoAI is designed to assist healthcare professionals and researchers in early detection and risk assessment.

---

## 🚀 Features

- 🔍 Early diabetes detection using clinical input data
- 🎯 Ensemble learning with Random Forest & Gradient Boosting
- ⚖️ SMOTE-based class balancing
- 🧪 Feature selection via Recursive Feature Elimination (RFE)
- 🧠 Explainable predictions with SHAP visualizations
- 📊 Real-time prediction dashboard via Streamlit

---

## 📁 Project Structure

```
glucoai/
├── glucoai.py           # Main Streamlit app
├── diabetes.csv         # Dataset (Pima Indians Diabetes)
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── visualizations.py    # Optional: plots and EDA
```

---

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Chakrasai/gluco.git
cd glucoai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn streamlit imbalanced-learn shap matplotlib seaborn
```

### 3. Run the Application

```bash
streamlit run glucoai.py
```

---

## 🧪 Sample Prediction

**Non-Diabetic Example Input:**

```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
1,85,66,29,0,26.6,0.351,31
```

**Expected Output:** 🟢 Non-Diabetic

---

## 📊 Visualizations

- Class Distribution
- Correlation Heatmap
- Boxplots per feature
- SHAP Summary and Force Plots

**Example code (in `glucoai.py` or `visualizations.py`):**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Outcome', data=df)
plt.title("Class Distribution")
plt.show()
```

---

## 📚 Dataset

- **Source:** [Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 female patients (aged 21+)
- **Features:** 8 clinical features + 1 output (Outcome)

---

## 📈 Model Performance

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Random Forest        | 87.3%    | 0.85      | 0.83   | 0.84     |
| Gradient Boosting    | 85.4%    | 0.83      | 0.80   | 0.82     |
| GlucoAI (Ensemble)   | 89.0%    | 0.87      | 0.85   | 0.86     |

---

## 🔬 Explainability (SHAP)

- SHAP bar plot (global importance)
- SHAP force plot (per-patient interpretation)
- Helps clinicians trust and interpret predictions

**Example:**

```python
import shap
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## 📦 requirements.txt

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
streamlit
shap
```

---

## 🤝 Contributions

Contributions are welcome!

1. Fork this repo
2. Create a feature branch
3. Push your changes
4. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Authors

- A. Chakrasai – chakrasaiaku@gmail.com
- K. Arun
- G. Karthik

*Institute of Aeronautical Engineering, Hyderabad – Dept. of CSE (AI & ML)*