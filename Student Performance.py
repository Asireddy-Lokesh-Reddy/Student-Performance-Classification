# %% [markdown]
# # Case Study 4: Predicting Student Dropout and Academic Success
# 
# ## Introduction
# This case study analyzes student data to understand academic outcomes and predict whether a student will Dropout, remain Enrolled, or Graduate. Pandas and NumPy are used for data analysis, Matplotlib and Seaborn for visualization, and Scikit-learn for classification.
# 
# ## Objectives
# - Understand and clean the dataset.
# - Perform exploratory data analysis.
# - Apply hypothesis testing to first-semester grades.
# - Build a multiclass logistic regression model.
# - Evaluate the model and interpret its results.

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ## A. Data Understanding and Preprocessing
# 
# In this section, the dataset is loaded into a Pandas DataFrame. The data structure, number of records, column types, missing values, duplicate records, and target categories are examined to understand the dataset before analysis.

# %%
df=pd.read_csv("Studentdata.csv",sep=";")

# %%
df.head()

# %%
df.shape

# %%
df.dtypes

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.nunique()

# %%
df.columns

# %%
print(df["Target"].value_counts())
print(df["Target"].value_counts(normalize=True).mul(100).round(2))

# %% [markdown]
# ### Data Cleaning
# 
# Some column names are renamed to remove unwanted spaces and special characters. This makes the column names easier to use in Python code. Missing values and duplicate records are checked before proceeding.

# %%
df.duplicated().sum()

# %%
df=df.rename(columns={'Daytime/evening attendance\t':'Daytime/evening attendance',"Mother's qualification":'Mother_s qualification',
                    "Father's qualification":'Father_s qualification',"Mother's occupation":'Mother_s occupation', 
                    "Father's occupation":'Father_s occupation'})

# %%
print("Target Variable : Target")
print(df['Target'].value_counts())
predictors=['Marital status',
 'Application mode',
 'Application order',
 'Course',
 'Daytime/evening attendance',
 'Previous qualification',
 'Previous qualification (grade)',
 'Nacionality',
 'Mother_s qualification',
 'Father_s qualification',
 'Mother_s occupation',
 'Father_s occupation',
 'Admission grade',
 'Displaced',
 'Educational special needs',
 'Debtor',
 'Tuition fees up to date',
 'Gender',
 'Scholarship holder',
 'Age at enrollment',
 'International',
 'Curricular units 1st sem (credited)',
 'Curricular units 1st sem (enrolled)',
 'Curricular units 1st sem (evaluations)',
 'Curricular units 1st sem (approved)',
 'Curricular units 1st sem (grade)',
 'Curricular units 1st sem (without evaluations)',
 'Curricular units 2nd sem (credited)',
 'Curricular units 2nd sem (enrolled)',
 'Curricular units 2nd sem (evaluations)',
 'Curricular units 2nd sem (approved)',
 'Curricular units 2nd sem (grade)',
 'Curricular units 2nd sem (without evaluations)',
 'Unemployment rate',
 'Inflation rate',
 'GDP']
print("Predictors:",predictors)

# %%
df.columns.tolist()

# %% [markdown]
# ### Classification of Attributes
# 
# The dataset attributes are grouped into numerical, binary, and categorical variables. Numerical variables represent measurable quantities, binary variables have two possible categories, and categorical variables represent groups or labels. Descriptive statistics are used to understand the numerical variables.

# %%
numerical=[
'Previous qualification (grade)','Age at enrollment',
 'Admission grade','Curricular units 1st sem (credited)',
 'Curricular units 1st sem (enrolled)',
 'Curricular units 1st sem (evaluations)',
 'Curricular units 1st sem (approved)',
 'Curricular units 1st sem (grade)',
 'Curricular units 1st sem (without evaluations)',
 'Curricular units 2nd sem (credited)',
 'Curricular units 2nd sem (enrolled)',
 'Curricular units 2nd sem (evaluations)',
 'Curricular units 2nd sem (approved)',
 'Curricular units 2nd sem (grade)',
 'Curricular units 2nd sem (without evaluations)',
 'Unemployment rate',
 'Inflation rate',
 'GDP']
df[numerical].nunique()

# %%
binary=['Daytime/evening attendance','Displaced',
 'Educational special needs',
 'Debtor',
 'Tuition fees up to date',
 'Gender',
 'Scholarship holder','International']
df[binary].nunique()

# %%
catogerical=['Marital status',
 'Application mode',
 'Application order',
 'Course','Previous qualification',
 'Nacionality',
 'Mother_s qualification',
 'Father_s qualification',
 'Mother_s occupation', 'Father_s occupation']
df[catogerical].nunique()

# %%
df[numerical].describe()

# %% [markdown]
# ### Outlier Detection
# 
# The Interquartile Range (IQR) method is used to identify possible outliers in numerical attributes. Values outside the lower and upper IQR limits are flagged for investigation. These values are not automatically removed because they may represent genuine student records.

# %%
for col in numerical :
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    o=df[(df[col]<q1-1.5*iqr) |
                  (df[col]>q3+1.5*iqr)]
    print(col,":",len(o),"Possible outlier")

# %% [markdown]
# ## B. Exploratory Data Analysis
# 
# Exploratory data analysis is performed to understand student characteristics and the distribution of academic outcomes. Histograms, bar charts, and box plots are used to identify patterns in age, admission grades, first-semester grades, and student outcomes.

# %%
import matplotlib.pyplot as plt 

# %%
s=['Age at enrollment',
 'Admission grade']
for col in s :
    df[col].hist()
    plt.title("Distribution of "+ col)
    plt.xlabel(col)
    plt.ylabel("number of students")
    plt.show()

# %% [markdown]
# ### Distribution of Student Outcomes
# 
# The charts show the number and percentage of students in the Dropout, Enrolled, and Graduate categories. Comparing these categories helps identify class imbalance in the dataset.

# %%
df["Target"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Target values Distribution")
plt.ylabel("")
plt.show()

# %%
df["Target"].value_counts()

# %%
df[numerical].describe()

# %%


# %% [markdown]
# ### Comparison of Student Characteristics
# 
# Box plots are used to compare admission grades, first-semester grades, and age at enrollment across the three outcome groups. The plots show differences in medians, distributions, and possible outliers. These patterns indicate associations and do not prove causation.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x=df['Target'], y=df['Admission grade'])
plt.title('Admission Grades by Student Outcome')
plt.xlabel('Student Outcome')
plt.ylabel('Admission Grade')
plt.show()

# %%
sns.boxplot(x=df['Target'],y=df['Curricular units 1st sem (grade)'])
plt.title('First-Semester Grades by Student Outcome')
plt.xlabel('Student Outcome')
plt.ylabel('First-Semester Grade')
plt.show()

# %%
sns.boxplot(data=df, x='Target', y='Age at enrollment')
plt.title('Age vs Student Outcome')
plt.xlabel("Student Outcome")
plt.ylabel("Age at Enrollment")
plt.show()

# %%

# 1. Bar chart
df["Target"].value_counts().plot(kind="bar")
plt.title("Student Outcomes")
plt.xlabel("Outcome")
plt.ylabel("Number of Students")
plt.show()

# 2. Histogram of admission grades
df["Admission grade"].plot(kind="hist", bins=20)
plt.title("Admission Grade Distribution")
plt.xlabel("Admission Grade")
plt.ylabel("Frequency")
plt.show()

# 3. Histogram of student ages
df["Age at enrollment"].plot(kind="hist", bins=20)
plt.title("Age Distribution")
plt.xlabel("Age at Enrollment")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# ## C. Hypothesis Testing
# 
# The purpose of hypothesis testing is to determine whether first-semester grades differ among Dropout, Enrolled, and Graduate students.
# 
# **Null hypothesis (H₀):** The mean first-semester grades are equal across all three outcome groups.
# 
# **Alternative hypothesis (H₁):** At least one group has a different mean first-semester grade.
# 
# The significance level is 0.05. If the p-value is below 0.05, the null hypothesis is rejected. Otherwise, there is insufficient evidence to reject it.

# %%
print("H0: Mean grades are equal across all student outcomes.")
print("H1: At least one group has a different mean grade.")

# %%
print("Selected Test: One-Way ANOVA")
print("Significance Level: 0.05")
print("Reason: Compare first-semester grades across three outcomes.")

# %% [markdown]
# ### Interpretation of Hypothesis Test
# 
# The F-statistic and p-value are used to assess whether the group means differ statistically. A significant result indicates that at least one group mean differs, but further testing is needed to determine which groups differ. Statistical significance does not necessarily imply practical importance.

# %%
import pandas as pd
from statsmodels.stats.oneway import anova_oneway

col = "Curricular units 1st sem (grade)"
groups = [df[df["Target"] == g][col].dropna()
          for g in ["Dropout", "Enrolled", "Graduate"]]

result = anova_oneway(groups)

print("F-statistic:", round(result.statistic, 2))
print("P-value:", result.pvalue)

if result.pvalue < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

# %% [markdown]
# ## D. Student Outcome Classification
# 
# A multiclass logistic regression model is developed to predict Dropout, Enrolled, and Graduate outcomes. Selected student characteristics are used as input features. The dataset is divided into training and testing sets, and the model is trained on the training data before making predictions on the test data.

# %%
# Part D: Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay,precision_score,recall_score,f1_score)

# picking the columns I want to use as inputs
features=["Admission grade","Previous qualification (grade)",
          "Age at enrollment","Application mode",
          "Application order","Course",
          "Previous qualification","Scholarship holder",
          "Tuition fees up to date","Debtor","Gender",
          "Displaced","Curricular units 1st sem (approved)",
          "Curricular units 1st sem (grade)","Curricular units 1st sem (evaluations)",
          "Curricular units 1st sem (credited)","Curricular units 1st sem (enrolled)"]

X=df[features]
y=df["Target"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# separating numeric and categorical columns, since they need different preprocessing

num_cols=["Admission grade",
          "Previous qualification (grade)",
          "Age at enrollment","Curricular units 1st sem (approved)",
          "Curricular units 1st sem (grade)","Curricular units 1st sem (evaluations)",
          "Curricular units 1st sem (credited)","Curricular units 1st sem (enrolled)"]

cat_cols=[c for c in features if c not in num_cols]

# scaling the numeric columns
# fit only on train so no info leaks from the test set
scaler=StandardScaler()
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])

# one-hot encoding the categorical columns
encoder=OneHotEncoder(handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols]).toarray()
X_test_cat=encoder.transform(X_test[cat_cols]).toarray()

# putting the numeric and categorical parts back together
X_train_final=np.concatenate((X_train_num,X_train_cat),axis=1)
X_test_final=np.concatenate((X_test_num,X_test_cat),axis=1)

# training the logistic regression
# class_weight="balanced" because the classes are not evenly distributed
model=LogisticRegression(max_iter=2000,class_weight="balanced")
model.fit(X_train_final,y_train)

# making predictions on the test set
y_pred=model.predict(X_test_final)

# confusion matrix to see where the model gets confused
labels=["Dropout","Enrolled","Graduate"]
cm=confusion_matrix(y_test,y_pred,labels=labels)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Model Evaluation
# 
# The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. These measures help assess overall performance and identify which outcome categories are predicted more or less accurately. Macro-averaged metrics give equal weight to each class.

# %%

# evaluating the model
print("Accuracy:",round(accuracy_score(y_test,y_pred),3))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

# macro averages treat each class equally
print("Macro Precision:",round(precision_score(y_test,y_pred,average="macro"),3))
print("Macro Recall:",round(recall_score(y_test,y_pred,average="macro"),3))
print("Macro F1-score:",round(f1_score(y_test,y_pred,average="macro"),3))


# %% [markdown]
# ### Interpretation of Logistic Regression Coefficients
# 
# The model coefficients indicate how input features are associated with each outcome in the fitted logistic regression model. Positive and negative coefficients indicate the direction of association relative to the model's class-specific decision function. Coefficients do not establish causation, and their interpretation depends on feature encoding and scaling.

# %%
# Part D.5: Model interpretation
import pandas as pd

# get feature names
num_names = list(num_cols)
cat_names = list(encoder.get_feature_names_out(cat_cols))
all_names = num_names + cat_names

# loop through each outcome and show top 10 coefficients
for i in range(3):
    print("\nOutcome:", model.classes_[i])
    coef = model.coef_[i]
    result = pd.DataFrame({"Feature": all_names, "Coefficient": coef})
    result["Absolute Value"] = abs(result["Coefficient"])
    result = result.sort_values("Absolute Value", ascending=False)
    print(result.head(10))

# %%


# %% [markdown]
# ## E. Conclusion
# 
# This case study uses data preprocessing, exploratory data analysis, hypothesis testing, and logistic regression to investigate student academic outcomes. The evaluation metrics help identify the strengths and weaknesses of the model. The results may support student guidance and academic assistance, but predictions should be validated and used responsibly.

# %%



