# 🧠 Intelligent Disease and Risk Prediction

This project implements a **smart disease prediction system** using **Machine Learning algorithms** to identify potential health risks based on symptoms provided by the user. It employs multiple classifiers and ensemble learning to provide accurate and robust predictions.

---

## 📌 Features

- Implements **Random Forest**, **SVM**, and **Gaussian Naive Bayes** classifiers
- Handles imbalanced data using `RandomOverSampler`
- Uses **Stratified K-Fold Cross-Validation** for model evaluation
- Visualizes model performance with **confusion matrices**
- Combines predictions using **majority voting** for enhanced accuracy
- Provides a function to predict diseases from **user-entered symptoms**

---

## 🛠️ Libraries Used

```python
numpy, pandas, seaborn, matplotlib
scikit-learn, imbalanced-learn
```

---

## 🔍 Model Evaluation

- **SVM Accuracy**: ~60.53%
- **Naive Bayes Accuracy**: ~37.98%
- **Random Forest Accuracy**: ~68.98%
- **Ensemble Model Accuracy**: **60.64%**

All models are evaluated using cross-validation and confusion matrices.

---

## 🧠 Prediction Example

```python
predict_disease("Itching,Skin Rash,Nodal Skin Eruptions")
```

**Output**:

```json
{
  "Random Forest Prediction": "Heart Attack",
  "Naive Bayes Prediction": "Urinary tract Infection",
  "SVM Prediction": "Impetigo",
  "Final Prediction": "Heart Attack"
}
```

---

## 🔮 Function: `predict_disease(input_symptoms)`

- Accepts a **comma-separated** string of symptoms
- Converts input into a **binary feature vector**
- Applies all trained models to the input
- Returns:
  - Individual predictions
  - Final prediction using **majority vote**

---

## 📊 Confusion Matrices

Confusion matrices help visualize model performance. Heatmaps are plotted for:

- SVM Classifier
- Naive Bayes Classifier
- Random Forest Classifier
- Combined Ensemble Model

These show prediction accuracy and misclassifications.

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
