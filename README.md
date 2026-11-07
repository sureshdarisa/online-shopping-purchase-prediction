# E-commerce Customer Purchase Intention Prediction

## 1. Project overview

This project applies machine learning classification algorithms to predict whether an online shopping session results in a purchase.

**Target:** `Revenue`
- `True` / `1`: the session resulted in a purchase.
- `False` / `0`: the session did not result in a purchase.

The project compares five classification algorithms and includes tuned versions of Logistic Regression and Decision Tree. A Streamlit application loads the saved model pipelines and evaluates them on an uploaded labelled CSV file. The application does not retrain models.

## 2. Dataset

**Dataset:** [Online Shoppers Purchasing Intention Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

- 12,330 original sessions.
- 17 input features and one binary target.
- 125 exact duplicate rows were removed, leaving 12,205 rows.
- The cleaned target distribution is imbalanced: 10,297 non-purchase sessions and 1,908 purchase sessions.

The 17 input features are:

`Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, and `Weekend`.

## 3. Methodology

1. Loaded the dataset and examined its dimensions, data types, missing values and target distribution.
2. Removed exact duplicate records.
3. Performed exploratory data analysis, including target distribution, feature distributions and relationships with the target.
4. Split the cleaned dataset into training and test sets using an 80:20 stratified split (`random_state=42`).
5. Used a preprocessing pipeline:
   - `StandardScaler` for numerical features.
   - `OneHotEncoder(handle_unknown="ignore")` for nominal categorical features.
   - Passthrough for the binary `Weekend` feature.
6. Trained five classifiers using the same training and test split.
7. Evaluated the models using Accuracy, ROC-AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).
8. Tuned Decision Tree and Logistic Regression using stratified cross-validation, optimising F1-score.
9. Saved complete preprocessing-and-model pipelines as Joblib files.
10. Built a Streamlit application to load and evaluate the saved models on an uploaded labelled test CSV.

## 4. Baseline model comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8873 | 0.8998 | 0.7536 | 0.4162 | 0.5363 | 0.5055 |
| Decision Tree | 0.8914 | 0.8808 | 0.6667 | 0.6126 | 0.6385 | 0.5754 |
| KNN | 0.8742 | 0.8002 | 0.6682 | 0.3901 | 0.4926 | 0.4465 |
| Gaussian Naive Bayes | 0.2773 | 0.7508 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| SVM | 0.8955 | 0.8738 | 0.7378 | 0.5157 | 0.6071 | 0.5607 |

## 5. Tuned model results

### Tuned Decision Tree

Best parameters:
- Criterion: `gini`
- Maximum depth: `4`
- Minimum samples per leaf: `10`
- Minimum samples split: `2`
- Class weight: `None`

Best cross-validation F1-score: `0.6441`

| Metric | Test score |
|---|---:|
| Accuracy | 0.9021 |
| AUC | 0.9237 |
| Precision | 0.7581 |
| Recall | 0.5497 |
| F1 | 0.6373 |
| MCC | 0.5924 |

### Tuned Logistic Regression

Best parameters:
- C: `0.1`
- Class weight: `balanced`
- Penalty: `l1`
- Solver: `liblinear`

Best cross-validation F1-score: `0.6209`

| Metric | Test score |
|---|---:|
| Accuracy | 0.8529 |
| AUC | 0.9134 |
| Precision | 0.5193 |
| Recall | 0.8089 |
| F1 | 0.6325 |
| MCC | 0.5670 |

## 6. Observations and conclusion

- The dataset is imbalanced, so accuracy alone is not sufficient to compare classifiers. F1-score, recall, AUC and MCC provide additional insight.
- The baseline Decision Tree achieved the highest baseline F1-score (0.6385).
- The tuned Decision Tree achieved the strongest overall combination among the evaluated models, with test accuracy 0.9021, AUC 0.9237 and MCC 0.5924.
- Tuned Logistic Regression achieved higher purchase recall (0.8089), meaning it detected more actual purchase sessions, but its precision was lower (0.5193), resulting in more false-positive purchase predictions.
- Gaussian Naive Bayes achieved very high purchase recall but generated many false positives at the default classification threshold.
- `PageValues` was strongly associated with the target. A sensitivity experiment without this feature showed a substantial decline in performance. Its usefulness depends on when a prediction is intended to be made: it may not be available for an early-session prediction. This is a potential deployment limitation, not proof of data leakage.

**Overall conclusion:** The tuned Decision Tree is the preferred general-purpose model based on the test results and MCC. Tuned Logistic Regression may be preferred in a use case where identifying as many potential purchasers as possible is more important than limiting false positives.

## 7. Streamlit application

The application:
- Loads saved Joblib pipelines from the `models/` folder.
- Accepts a labelled CSV containing the 17 features and `Revenue`.
- Allows the user to select one of the seven saved models.
- Displays Accuracy, AUC, Precision, Recall, F1-score and MCC.
- Displays a confusion matrix and classification report.
- Provides a downloadable CSV containing predictions.

### Run locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure all seven `.joblib` files are present in the `models/` directory.
5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Upload `test_data.csv` and select a model.

**Important:** Scikit-learn models saved with Joblib should be loaded with a compatible scikit-learn version. For reproducibility, use the same scikit-learn version that was used to train and save the models. Update `requirements.txt` to pin that exact version.

## 8. Repository structure

```text
online-shopping-purchase-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── test_data.csv
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_nb.joblib
    ├── svm.joblib
    ├── tuned_decision_tree.joblib
    └── tuned_logistic_regression.joblib
```

## 9. Author and repository

- **Author:** Add your name here.
- **GitHub repository:** Add your repository URL here.
