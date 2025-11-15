# 🏥 Personalized Healthcare Recommendation System  
**🔗 Live Deployment:**  
👉 **(https://personalized-healthcare-recommendations-by-isaar.streamlit.app/)**  

An end-to-end machine learning solution designed to **predict heart disease risk** and deliver **personalized healthcare recommendations** through an interactive **Streamlit web application**.

---

## 👤 Developer  
**Mohd Isaar**  
Data Analyst Intern — Unified Mentor Pvt. Ltd.

---

## 📌 Project Overview

This project analyzes cardiovascular health data from **cleaned_merged_heart_dataset.csv**, including attributes such as:

- Age  
- Blood Pressure  
- Cholesterol  
- Chest Pain Type  
- Maximum Heart Rate  
- ECG Results  
- Exercise-Induced Angina  
- Other clinical indicators  

### 🔧 ML Pipeline Includes:
- Data Cleaning & Transformation  
- Exploratory Data Analysis (EDA)  
- Feature Scaling  
- Logistic Regression Model  
- Model Export using `joblib`  
- Deployment using **Streamlit**  

### 🧠 Final Model  
- **Algorithm:** Logistic Regression  
- **Accuracy:** 72%  
- **Why:** Interpretable, reliable, and well-suited for healthcare classification tasks  

---

## 📁 Folder Structure

```
Data/
└── cleaned_merged_heart_dataset.csv

Note book/
└── EDA_and_Model.ipynb

Documentation/
└── Project_Report.pdf

app.py

healthcare_recommendation_model.pkl
heart_model.pkl
scaler.pkl

requirements.txt
```

---

## 🛠 Tools & Technologies

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## 🚀 How to Run the App

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 📈 Model Summary

| Component  | Details |
|-----------|---------|
| Algorithm | Logistic Regression |
| Accuracy  | 72% |
| Strengths | Interpretable, efficient, deployment-ready |

---

## 🔮 Future Enhancements

- Cloud deployment (AWS / Azure)  
- Multi-disease prediction capability  
- Integration with real-time patient data  
- Advanced ML models (Random Forest, XGBoost)  

---

## ✅ Conclusion

This project delivers a full-stack ML-powered healthcare prediction system that identifies heart disease risk and provides actionable, personalized recommendations—supporting early detection, patient awareness, and informed clinical decisions.
