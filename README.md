# Predicting Student Dropout and Academic Success Using Data Science

## Overview

This project analyzes student data and builds a multiclass machine learning model to predict a student's academic outcome as one of the following:

- **Dropout**
- **Enrolled**
- **Graduate**

The project uses Python for data preprocessing, exploratory data analysis (EDA), statistical hypothesis testing, classification, and model evaluation. The results can help identify patterns associated with student outcomes and inform further investigation into student-support initiatives.

> **Important:** The model identifies statistical associations and predicts outcomes; it does not establish that any feature causes a student to drop out. Predictions should support, not replace, human judgement.

## Objectives

- Understand and preprocess the student dataset.
- Check data types, missing values, duplicates, and unusual values.
- Analyze the distribution of student outcomes and selected numerical attributes.
- Visualize relationships between student characteristics and academic outcomes.
- Use one-way ANOVA to test whether first-semester grades differ across outcome groups.
- Train a multiclass Logistic Regression model.
- Evaluate the model using accuracy, precision, recall, F1-score, macro averages, and a confusion matrix.
- Interpret model coefficients and discuss limitations and responsible use.

## Project Structure

A suggested GitHub repository structure is:

```text
student-dropout-prediction/
├── Student Performance(1).py
├── Student Performance(20261025-143318).ipynb
├── Studentdata.csv
├── student performance(3).docx
└── README.md
```

You may rename files to simpler names before uploading, for example `student_performance.py`, `student_performance.ipynb`, `student_data.csv`, and `case_study.docx`. Update the dataset path in the code if you rename the CSV file.

## Technologies Used

- **Python**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Matplotlib** and **Seaborn** — data visualization
- **Scikit-learn** — preprocessing, Logistic Regression, and evaluation
- **Statsmodels** — one-way ANOVA
- **Jupyter Notebook** — interactive analysis

## Dataset

The project uses a student dataset containing information such as:

- Demographic and background attributes
- Previous qualifications and admission grades
- Enrollment and financial-status indicators
- First- and second-semester academic information
- Economic indicators
- The target variable, `Target`, with the classes `Dropout`, `Enrolled`, and `Graduate`

The notebook loads the CSV using a semicolon delimiter:

```python
df = pd.read_csv("Studentdata.csv", sep=";")
```

Place the dataset in the expected location, or change the path in the notebook/script to match your folder structure. Include the dataset in your repository only if its license and distribution terms permit it. Otherwise, add instructions for obtaining it from the original source.

## Methodology

### 1. Data Understanding and Preprocessing

The analysis inspects the first records, dataset dimensions, column data types, missing values, unique values, duplicate records, and target-class distribution. Some column names are renamed for easier use in Python.

Numerical, binary, and categorical attributes are then identified. The Interquartile Range (IQR) method is used to flag possible outliers for investigation; outliers are not automatically removed.

### 2. Exploratory Data Analysis

The notebook explores the dataset using:

- Bar and pie charts showing the distribution of `Target`
- Histograms for age at enrollment and admission grade
- Box plots comparing admission grades, first-semester grades, and age across outcome groups
- Descriptive statistics for selected numerical attributes

These visualizations help describe patterns and possible differences among the outcome groups.

### 3. Hypothesis Testing

A one-way ANOVA is used to investigate whether mean first-semester grades differ among Dropout, Enrolled, and Graduate students.

- **Null hypothesis (H₀):** The mean first-semester grades are equal across all three outcome groups.
- **Alternative hypothesis (H₁):** At least one group has a different mean first-semester grade.
- **Significance level:** 0.05

The p-value is compared with 0.05 to decide whether to reject the null hypothesis. ANOVA assumptions should also be considered when interpreting the result. A significant result does not, by itself, show which groups differ or whether the difference is practically important.

### 4. Classification Model

The notebook builds a multiclass Logistic Regression classifier.

1. Select predictor features and the `Target` label.
2. Split the data into training and testing sets using an 80/20 split, with stratification and `random_state=42`.
3. Standardize numerical features using `StandardScaler`, fitting it on the training set only.
4. One-hot encode categorical features using `OneHotEncoder(handle_unknown="ignore")`.
5. Train `LogisticRegression(max_iter=2000, class_weight="balanced")`.
6. Predict outcomes for the test set.

**Important modelling consideration:** The selected features include first-semester academic results, while the target is an eventual outcome. This may be useful for a later-stage prediction, but it can introduce target-timing leakage if the intended use is to predict dropout before those results are available. For an early-warning system, only use features that would genuinely be available at the intended prediction time.

### 5. Model Evaluation and Interpretation

The notebook evaluates predictions with:

- Accuracy
- Per-class precision, recall, and F1-score
- Macro precision, macro recall, and macro F1-score
- Confusion matrix
- Logistic Regression coefficients

Macro-averaged metrics give each class equal weight and are useful when the target classes are imbalanced. Review per-class results rather than relying on accuracy alone. Coefficients show model associations under the chosen encoding and scaling; they do not establish causation.

## Installation

Use Python 3 and install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
```

## How to Run

1. Clone or download this repository.
2. Make sure the dataset file is in the expected location and is named `Studentdata.csv`, or update the file path in the code.
3. Open a terminal in the project folder.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Student Performance(20261025-143318).ipynb`.
6. Run the cells in order.

You can also run the Python script from the terminal:

```bash
python "Student Performance(1).py"
```

Ensure the CSV path is correct before running the script. The script and notebook should be checked for consistent file paths and successful execution in your environment.

## Results

The notebook produces visualizations, the ANOVA F-statistic and p-value, a confusion matrix, classification metrics, and the largest Logistic Regression coefficients for each outcome class.

**Add your actual results here after running the notebook.** Include the measured test accuracy, macro F1-score, and per-class precision/recall/F1. Do not report an expected or target accuracy as if it were measured performance.

## Limitations

- The quality of predictions depends on dataset quality, representativeness, and feature availability.
- Class imbalance can make overall accuracy misleading.
- Using semester results to predict an eventual outcome may not be suitable for very early intervention.
- Logistic Regression models relationships within its modelling assumptions and may not capture every nonlinear pattern.
- ANOVA conclusions depend on its assumptions and do not imply causation.
- The model should be evaluated for fairness and validated on appropriate data before real-world use.

## Responsible Use

Use predictions as one signal for offering additional academic guidance or support—not to penalize students, deny opportunities, or label individuals as failures. Review model performance across groups, protect student data, and involve appropriate staff in decisions.

## Future Improvements

- Compare Logistic Regression with other suitable classification algorithms.
- Tune hyperparameters using cross-validation without using the test set for model selection.
- Investigate class-specific errors and calibration.
- Build an early-warning model using only information available at the intended prediction time.
- Validate the model on separate or more recent data and assess fairness across relevant groups.

## Author

Add your name, institution, course, and GitHub profile here.

## License

Add a license if you want others to reuse this project. Confirm the dataset's license separately before redistributing the data.
