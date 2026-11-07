# E-commerce Customer Purchase Intention Prediction Using Machine Learning

## 1. Project Overview

The objective of this project is to develop and evaluate machine learning classification models to predict whether an online shopping session will result in a purchase.

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

The target variable, `Revenue`, indicates whether a visitor completed a purchase during a browsing session:

- `True` – Purchase completed.
- `False` – No purchase completed.

Five machine learning classification algorithms are implemented and compared using multiple evaluation metrics.

## 2. Project Links

- **GitHub Repository:** [View Source Code](hhttps://github.com/sureshdarisa/online-shopping-purchase-prediction)
- **Live Streamlit Application:** [Open Streamlit App](hhttps://online-shopping-purchase-prediction-gugbhpuym5fjqwl9kvcwdb.streamlit.app/)

## 3. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

| Description | Details |
|---|---|
| Original records | 12,330 |
| Records after duplicate removal | 12,205 |
| Input features | 17 |
| Target variable | Revenue |
| Problem type | Binary Classification |
| Missing values | None identified |

### Features

The dataset contains information about visitors' browsing behaviour and shopping sessions.

Some important features include:

- Administrative and Administrative_Duration
- Informational and Informational_Duration
- ProductRelated and ProductRelated_Duration
- BounceRates and ExitRates
- PageValues
- SpecialDay
- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType
- Weekend

### Data Preprocessing

The following preprocessing activities were performed:

1. Inspected the dataset for missing values.
2. Identified and removed 125 duplicate records.
3. Separated the independent variables and target variable.
4. Converted categorical variables using One-Hot Encoding.
5. Applied StandardScaler to numerical features.
6. Used a stratified 80:20 train-test split.

The cleaned dataset contains 12,205 records.

The target distribution is:

| Revenue | Number of records | Percentage |
|---|---:|---:|
| False (No Purchase) | 10,297 | 84.37% |
| True (Purchase) | 1,908 | 15.63% |

The dataset is imbalanced, with non-purchase sessions representing the majority class.

## 4. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models were trained and evaluated using the same training and testing datasets.

### Evaluation Metrics

The following metrics were used:

- **Accuracy:** Measures the proportion of correctly classified observations.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures how many predicted purchases were actual purchases.
- **Recall:** Measures how many actual purchases were correctly identified.
- **F1-score:** Harmonic mean of Precision and Recall.
- **MCC:** Matthews Correlation Coefficient, which considers all four confusion-matrix outcomes.

## 5. Model Performance Comparison

The following table presents the evaluation results obtained from the test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8873 | **0.8998** | **0.7536** | 0.4162 | 0.5363 | 0.5055 |
| Decision Tree | 0.8914 | 0.8808 | 0.6667 | **0.6126** | **0.6385** | **0.5754** |
| KNN | 0.8742 | 0.8002 | 0.6682 | 0.3901 | 0.4926 | 0.4465 |
| Gaussian Naive Bayes | 0.2773 | 0.7508 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| SVM | **0.8955** | 0.8738 | 0.7378 | 0.5157 | 0.6071 | 0.5607 |

**Note:** Bold values indicate the highest score for each evaluation metric.

## 6. Observations

### Logistic Regression

- Achieved an accuracy of 88.73%.
- Obtained the highest AUC of 0.8998.
- Achieved the highest precision of 0.7536.
- Recall was relatively low at 0.4162.

**Observation:** Logistic Regression demonstrated strong class discrimination and precision, but it missed a considerable proportion of actual purchase sessions.

### Decision Tree

- Achieved an accuracy of 89.14%.
- Obtained the highest recall of 0.6126.
- Achieved the highest F1-score of 0.6385.
- Obtained the highest MCC of 0.5754.

**Observation:** Decision Tree provided the best balance between precision and recall. It was the strongest baseline model based on F1-score and MCC.

### K-Nearest Neighbors (KNN)

- Achieved an accuracy of 87.42%.
- Obtained an AUC of 0.8002.
- Recall was 0.3901.

**Observation:** KNN achieved reasonable accuracy but comparatively weaker purchase detection and class discrimination than the other models.

### Gaussian Naive Bayes

- Achieved an accuracy of 27.73%.
- Obtained a very high recall of 0.9791.
- Precision was only 0.1756.

**Observation:** Gaussian Naive Bayes identified almost all actual purchase sessions but incorrectly classified many non-purchase sessions as purchases, resulting in poor overall performance.

### Support Vector Machine (SVM)

- Achieved the highest accuracy of 89.55%.
- Obtained a precision of 0.7378.
- Recall was 0.5157.
- F1-score was 0.6071.

**Observation:** SVM demonstrated strong overall accuracy and a good balance between precision and recall, although its recall was lower than that of Decision Tree.

## 7. Additional Hyperparameter Tuning Experiments

Hyperparameter tuning was also performed for Decision Tree and Logistic Regression using GridSearchCV.

The objective was to improve model performance, particularly the F1-score, while considering the imbalance in the target variable.

### Tuned Decision Tree

Best parameters:

- Criterion: Gini
- Maximum depth: 4
- Minimum samples per leaf: 10
- Minimum samples split: 2

Test performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.9021 |
| AUC | 0.9237 |
| Precision | 0.7581 |
| Recall | 0.5497 |
| F1-score | 0.6373 |
| MCC | 0.5924 |

### Tuned Logistic Regression

Best parameters:

- C: 0.1
- Class weight: Balanced
- Penalty: L1
- Solver: liblinear

Test performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.8529 |
| AUC | 0.9134 |
| Precision | 0.5193 |
| Recall | 0.8089 |
| F1-score | 0.6325 |
| MCC | 0.5670 |

The tuned Logistic Regression model achieved considerably higher recall, identifying more actual purchasing customers, while the tuned Decision Tree achieved stronger accuracy, AUC and MCC.

## 8. Overall Conclusion

Based on the experimental results:

1. Decision Tree achieved the highest F1-score and MCC among the five baseline models.
2. Logistic Regression achieved the highest AUC and precision.
3. SVM achieved the highest accuracy.
4. KNN showed comparatively weaker classification performance.
5. Gaussian Naive Bayes achieved very high recall but suffered from a large number of false-positive predictions.

Since the dataset is imbalanced, accuracy alone is not sufficient to determine the best model.

Among the five baseline models, **Decision Tree is considered the most suitable model** based on its F1-score and MCC.

The additional tuning experiments demonstrated that model performance can be further improved depending on the evaluation metric and business objective.

## 9. Streamlit Application

A Streamlit application was developed to demonstrate the trained classification models.

The application supports:

- Uploading a test CSV file containing the required input features and target variable.
- Selecting one of the saved machine learning models.
- Loading the saved model without retraining.
- Evaluating the selected model on the uploaded dataset.
- Displaying Accuracy, AUC, Precision, Recall, F1-score and MCC.
- Displaying the confusion matrix and classification report.

**Live Application:** [Open Streamlit Application](https://online-shopping-purchase-prediction-gugbhpuym5fjqwl9kvcwdb.streamlit.app/)

## 10. Repository Structure

```text
online-shopping-purchase-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_nb.joblib
│   ├── svm.joblib
│   ├── tuned_decision_tree.joblib
│   └── tuned_logistic_regression.joblib
│
├── notebooks/
│   └── online_shoppers_analysis.ipynb
│
└── screenshots/
    └── streamlit_app.png
```

## 11. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

**Project:** E-commerce Customer Purchase Intention Prediction Using Machine Learning Classification Algorithms

**Dataset Source:** UCI Machine Learning Repository