# 📖 Machine Learning Study Guide: Chronic Kidney Disease Classification

Welcome to your in-depth study guide! If you know nothing about Machine Learning (ML), Exploratory Data Analysis (EDA), or the code written for this project, you are in the right place. 

This document will walk you through **what** we built, **why** we built it, and explain the code **line-by-line** (grouped by logical concepts).

---

## 🧠 Part 1: The Basics - What are we doing?

Imagine you are a doctor. A patient walks in, and you take their blood pressure, age, blood test results, etc. Based on these **measurements**, you diagnose whether they have a disease or not.

Machine Learning does exactly this, but using math and past data.
*   **The Data:** We have a spreadsheet (dataset) of 400 past patients.
*   **Features (Inputs):** The 24 measurements (age, blood pressure, hemoglobin, etc.) for each patient.
*   **Target (Output):** A label saying whether the patient actually had Chronic Kidney Disease (CKD) or Not (Not CKD).

**Our Goal:** To train an algorithm to look at the 24 features of *new* patients and accurately predict the target (CKD or Not CKD). This specific type of problem is called **Binary Classification** (because there are only two possible outcomes: Yes or No).

To achieve this, we follow a standard ML pipeline:
1.  **Data Preprocessing:** Cleaning up the messy spreadsheet.
2.  **Exploratory Data Analysis (EDA):** Making charts to understand the data.
3.  **Feature Engineering:** Deciding which measurements are actually useful.
4.  **Model Training & Evaluation:** Teaching the algorithm and testing how smart it is.
5.  **Deployment:** Making a web app so anyone can use the trained algorithm.

Let's look at the code for each step.

---

## 🧹 Part 2: Data Preprocessing (`src/data_preprocessing.py`)

Real-world data is extremely messy. There are missing values, typos, and text where numbers should be. Algorithms can **only understand numbers**. Preprocessing is the act of converting our messy spreadsheet into a clean grid of pure numbers.

### Step 2.1: Setup and Loading Data

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import joblib
import os
```
*   **What this does:** These lines import the "tools" we need. 
    *   `pandas` (usually imported as `pd`) is like Excel for Python. It lets us read and manipulate data in tables called DataFrames.
    *   `numpy` (as `np`) is for heavy math.
    *   `sklearn` (Scikit-Learn) is the ultimate ML library. It contains all our algorithms and preprocessing tools.

```python
# Constants defining our columns
NUMERIC_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
CATEGORICAL_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
TARGET_COL = "classification"
```
*   **What this does:** We explicitly list which columns contain numbers (e.g., `age`, `bp` for blood pressure) and which contain categories/text (e.g., `rbc` for red blood cells which can be "normal" or "abnormal").

### Step 2.2: Cleaning the Mess

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() # Make a copy so we don't destroy original data
    
    # Drop id column
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
```
*   **Why drop 'id'?** A patient's ID number has absolutely nothing to do with whether they have a disease. If we leave it in, the algorithm might get confused and think ID #44 causes kidney disease. We drop useless columns.

```python
    # Strip whitespace from all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Replace ? and empty strings with NaN
    df.replace({"?": np.nan, "": np.nan, "\t": np.nan}, inplace=True)
```
*   **The Mess:** Sometimes the raw data has hidden spaces like `"yes "` instead of `"yes"`. Sometimes a doctor didn't record a value and put a `?`.
*   **The Fix:** `.str.strip()` removes hidden spaces. `replace` turns weird symbols into `NaN` (Not a Number), which is Python's official way of saying "this data is missing".

```python
    # Fix numeric columns stored as object
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
```
*   **The Problem:** Sometimes a column like 'Packed Cell Volume' (`pcv`) was accidentally saved as text instead of numbers.
*   **The Fix:** `pd.to_numeric` forces the column to be recognized as numbers.

### Step 2.3: Imputing Missing Values (Guessing the Blanks)

Our dataset has about 30% missing data for some tests. ML models crash if you give them missing data (NaN). We have two choices: delete the patients with missing data (bad, we lose too much data), or **Impute** (guess/fill in) the blanks.

```python
def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    # --- Categorical: mode imputation ---
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
```
*   **Categorical Fix:** If a text feature like "appetite" is missing, we fill it with the **mode** (the most common answer in the whole dataset). 

```python
    # --- Numerical: KNN imputation ---
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])
```
*   **Numerical Fix:** For numbers like blood pressure, we use **KNN Imputation**. 
*   **What is KNN?** K-Nearest Neighbors. It finds the 5 patients (`n_neighbors=5`) who are most medically similar to our patient with the missing data, averages their blood pressure, and fills in the blank with that average. It's a very smart way to guess.

### Step 2.4: Encoding (Turning Text into Numbers)

```python
def encode_features(df: pd.DataFrame):
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
```
*   **The Problem:** ML models can't read the word "normal" or "abnormal". 
*   **The Fix:** `LabelEncoder` changes text to numbers. E.g., "abnormal" becomes `0`, "normal" becomes `1`. We save these "dictionaries" (`label_encoders`) so we can decode them later if needed.

### Step 2.5: Splitting and Scaling

```python
def split_and_scale(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(TARGET_COL, axis=1) # The 24 features (Inputs)
    y = df[TARGET_COL]              # The disease label (Output)

    # Split into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
```
*   **Train/Test Split:** You wouldn't give a student the answers to a final exam while they are taking it. Similarly, we split our 400 patients into a **Training Set (80%, 320 patients)** to teach the algorithm, and a **Testing Set (20%, 80 patients)** to test how well it learned. We hide the Test set during training.
*   `stratify=y`: Ensures the 80/20 ratio keeps the same percentage of sick/healthy people in both sets.

```python
    # Scale the numbers
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
```
*   **Scaling:** Imagine one feature is "Age" (ranges from 10 to 90) and another is "White Blood Cell Count" (ranges from 2,000 to 30,000). The algorithm might wrongly think White Blood Cells are 1000x more important just because the numbers are bigger.
*   **The Fix:** `StandardScaler` shrinks all numbers down to the same scale (mostly between -3 and 3), putting all features on a level playing field.

---

## 📊 Part 3: Exploratory Data Analysis (`src/eda.py`)

**EDA** is the process of creating charts to understand our data *before* we train a model. It helps us spot weird anomalies and understand which features are highly connected to the disease.

*The code in `eda.py` mostly uses `matplotlib` and `seaborn` (charting libraries) to draw boxes and lines. Let's focus on the concepts of the charts generated rather than the drawing code itself.*

1.  **Target Distribution (`plot_target_distribution`)**: A simple bar chart showing how many patients have CKD vs Not CKD. This tells us if our data is "imbalanced".
2.  **Missing Value Matrix (`plot_missing_heatmap`)**: A visual map showing where data was missing before we imputed it.
3.  **Numerical Distributions (`plot_numerical_distributions`)**: Creates "histograms". It shows, for example, the distribution of ages. Do sick people tend to be older? This chart shows us the overlap between healthy and sick patients for every number feature.
4.  **Correlation Heatmap (`plot_correlation_heatmap`)**: A massive color grid. If Hemoglobin goes down, does Packed Cell Volume go down too? This chart measures how features move together (Correlation). Highly correlated features might be redundant.
5.  **Box Plots**: Shows the median, quartiles, and "outliers" (extreme, weird values) for key features.

---

## 🔍 Part 4: Feature Engineering (`src/feature_engineering.py`)

Do we really need all 24 features? Maybe "appetite" is useless for predicting kidney disease, while "Serum Creatinine" is critical. Feeding bad/useless features to a model makes it confused (this is called the curse of dimensionality).

```python
def rf_feature_importance(X_train, y_train, feature_names):
    rf = RandomForestClassifier(...)
    rf.fit(X_train, y_train)
    return rf.feature_importances_
```
*   **Random Forest Importance:** We train a quick "Random Forest" model. This model is very good at telling us, mathematically, which features it relied on the most to make decisions. 

```python
def mutual_information_scores(X_train, y_train, feature_names):
    mi = mutual_info_classif(X_train, y_train, ...)
    return mi
```
*   **Mutual Information:** This is a statistics concept. It measures how much knowing Feature A reduces your uncertainty about the Target (CKD). High score = very informative feature.

*   **Combined Ranking:** In the code, we average the rankings from both methods to get a final, robust list of the "Most Important Features" (e.g., Hemoglobin, Specific Gravity). We could choose to drop the bottom 5 useless features to make our model faster and simpler, though in this project we kept them as algorithms handled them well.

---

## 🤖 Part 5: Model Training & Evaluation (`src/train.py`)

This is where the actual Machine Learning happens.

### Step 5.1: Defining the Models

```python
MODELS = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": { ... }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": { "n_estimators": [100, 200, 300], "max_depth": [5, 10, None] }
    },
    # ... (other models)
}
```
*   We define 5 different "Algorithms" (students). Think of them as different ways of solving a puzzle:
    *   **Logistic Regression:** Draws a straight mathematical line between sick and healthy.
    *   **Decision Tree:** Plays a game of 20 questions ("Is Age > 50? If yes, is Blood Pressure > 90?").
    *   **Random Forest:** Creates 100 different Decision Trees and makes them vote. (Usually the most powerful).
    *   **KNN:** Looks at the closest matching patients.
    *   **Gradient Boosting:** Builds trees sequentially, where each new tree tries to fix the mistakes of the previous one.

### Step 5.2: Hyperparameter Tuning (GridSearchCV)

```python
grid = GridSearchCV(config["model"], config["params"], cv=cv, scoring="f1")
grid.fit(X_train, y_train)
```
*   Algorithms have "settings" called **Hyperparameters** (e.g., how deep a Decision Tree should grow). 
*   **GridSearchCV** tries *every possible combination* of settings we gave it in `params`, tests them, and automatically picks the absolute best settings for that specific algorithm.
*   `cv=cv`: This is **Cross-Validation**. It prevents the model from just memorizing the training data by splitting the training data into 5 chunks, training on 4, testing on 1, and rotating.

### Step 5.3: Evaluation (The Final Exam)

Once the models are trained, we give them the hidden **Test Set** (`X_test`). They give us predictions (`y_pred`). We compare their predictions to the true answers (`y_test`).

```python
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
```
In a medical setting, evaluation is highly specific:
*   **Accuracy:** Overall, how many did it get right? (e.g., 98%)
*   **Confusion Matrix:** A 2x2 grid showing:
    *   **True Positives (TP):** Sick people correctly identified as Sick.
    *   **True Negatives (TN):** Healthy people correctly identified as Healthy.
    *   **False Positives (FP):** Healthy people wrongly told they are Sick (Scary, but they will do more tests).
    *   **False Negatives (FN):** Sick people wrongly told they are Healthy (EXTREMELY DANGEROUS).
*   **Recall (Sensitivity):** Out of all the *actually* sick people, how many did we catch? `TP / (TP + FN)`. **In medicine, we want Recall to be as close to 100% as possible so no sick patient is sent home.**
*   **Precision:** When the model *says* you are sick, how often is it right? `TP / (TP + FP)`. 

*The script compares all 5 models and saves the one with the highest scores as `best_model.pkl` (a file format that saves trained ML models).*

---

## 🌐 Part 6: Deployment (`app.py`)

A saved ML model (`.pkl` file) is useless to a doctor on its own. We use a framework called **Streamlit** to build a web interface around it.

```python
# Load the saved brains and tools
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
```
*   When the web app starts, it loads the winning model (Random Forest), and the exact `scaler` and `label_encoders` we used during training.

```python
age = st.number_input("Age (years)", 1, 120, 50)
rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
```
*   Streamlit functions like `st.number_input` automatically create nice looking web-forms for a user to type in a patient's details.

```python
if st.button("🔍 Predict CKD Status"):
    # 1. Gather all inputs into a dataframe
    # 2. Encode text like "abnormal" into numbers using our saved label_encoders
    # 3. Scale the numbers using our saved scaler
    # 4. Predict!
    prediction = model.predict(input_scaled)[0]
```
*   When the doctor clicks "Predict", the app takes their raw inputs, passes them through the exact same preprocessing pipeline (Step 2) to convert them into the pure scaled numbers the model expects, and asks the model for a prediction!

---

## 🎉 Summary
You took messy hospital records, cleaned them up, visualized the relationships, trained 5 advanced mathematical algorithms to detect patterns, evaluated them rigorously focusing on patient safety (Recall), and deployed the winning algorithm to a user-friendly web app. That is end-to-end Machine Learning!
