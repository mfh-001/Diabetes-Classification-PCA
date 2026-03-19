# Medical Diagnostic Classification & PCA Analysis

This repository implements a machine learning pipeline to predict diabetic outcomes using a clinical dataset. The project demonstrates the application of supervised learning (Logistic Regression) alongside unsupervised dimensionality reduction (PCA) to visualize high-dimensional medical data in a 2D space.

## Overview

The system processes health data to determine the probability of a patient being diabetic:
- **Binary Classification**: Implementing Logistic Regression to model the relationship between clinical features and patient outcomes.
- **Dimensionality Reduction**: Utilizing Principal Component Analysis (PCA) to project the 8-dimensional clinical space into a 2D plane.
- **Performance Evaluation**: Comprehensive scoring using Accuracy, Precision, Recall, F1-Score, and ROC-AUC analysis.

## Engineering Logic

### 1. Logistic Regression Model
The model predicts the probability $P$ of a patient being diabetic using the logistic (sigmoid) function. The core equation implemented is:

$$P(y=1|z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the linear combination of inputs: $z = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$. 
This maps any real-valued input into the range $[0, 1]$, representing the diagnostic confidence.



### 2. PCA: Dimensionality Reduction
The dataset contains 8 features (Glucose, Blood Pressure, BMI, etc.). To visualize feature separation, I applied PCA to find the directions of maximum variance. 
- **PC1**: Captures the highest possible variance in the data.
- **PC2**: Captures the second highest variance orthogonal to PC1.

In the notebook, the variance captured is visualized to show how well the top two components represent the full clinical profile.


## Tech Stack

- **Environment:** Google Colab / Jupyter Notebook
- **Language:** Python 3
- **Libraries:**
    - **Scikit-Learn:** `StandardScaler` for normalization, `LogisticRegression` for modeling, and `PCA` for reduction.
    - **Pandas/NumPy:** Data cleaning and matrix math.
    - **Seaborn/Matplotlib:** Generating ROC-AUC curves and PCA scatter plots.

## 📂 Project Structure

- **Diabetes_Classification_PCA.ipynb**: The full documented pipeline (Scaling -> Training -> Evaluation -> PCA).
- **dataset/diabetes_database.csv**: Clinical records containing 768 patient samples. [Available on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Performance Metrics

The project generates high-resolution evaluation plots:
1. **Confusion Matrix**: Visualizing True Positives vs. False Negatives.
2. **ROC-AUC Curve**: Measuring the model's ability to distinguish between classes (Area Under Curve).
3. **PCA Visualization**: A "shaded" scatter plot displaying correctly vs. incorrectly classified points in the reduced feature space.

<img width="988" height="542" alt="Screenshot 2026-01-01 at 7 39 57 PM" src="https://github.com/user-attachments/assets/0d106e3c-51e1-45e5-8f6e-507cb22d017e" />

## ⚠️ Disclaimer

This repository serves as a showcase of my technical growth and learning journey. The contents are intended strictly for educational and research purposes. All outputs should be treated as conceptual references rather than production-ready solutions.
