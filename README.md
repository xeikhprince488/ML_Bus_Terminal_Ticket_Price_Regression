# Bus Terminal Ticket Price Regression - Machine Learning Models Comparison

A comprehensive Jupyter Notebook demonstrating complete machine learning regression workflow using multiple algorithms to predict ticket prices in bus terminals. The project includes data preprocessing, dimensionality reduction with PCA, and comparative analysis of 4 different regression models.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Dataset Overview](#dataset-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Notebook Sections Explained](#notebook-sections-explained)
- [Preprocessing Steps](#preprocessing-steps)
- [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
- [Regression Models](#regression-models)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Results & Conclusions](#results--conclusions)
- [How to Use](#how-to-use)
- [Key Concepts](#key-concepts)
- [Future Enhancements](#future-enhancements)

## 🎯 Overview

This notebook demonstrates a complete regression machine learning pipeline using a Bus Terminal dataset. It predicts ticket prices based on various passenger and operational features. The project compares 4 different regression algorithms and evaluates their performance using multiple metrics.

**Objective:** Build and compare regression models to accurately predict ticket prices for bus terminal passengers.

**Target Variable:** `Predicted_Ticket_Price` (Continuous numeric values representing ticket price in currency units)

## 🔄 Project Workflow

```
Load Data → Explore Data → Clean Data → Encode Features → 
Analyze Correlations → Drop Weak Features → Feature Scaling → 
Dimensionality Reduction (PCA) → Train Multiple Models → 
Evaluate Models → Visualize Results → Compare & Conclude
```

## 📊 Dataset Overview

**File:** `bus_terminal_dataset_55000_50plus.csv`

**Size:** 55,000 records with 50+ features

**Content:** Passenger information and operational metrics from bus terminals

### Original Columns:
- `Passenger_ID` - Unique identifier (dropped)
- `Passenger_Name` - Passenger name (dropped)
- `Age` - Passenger age (dropped)
- `Gender` - Passenger gender (dropped)
- `Predicted_Ticket_Price` - **Target Variable** (Price to predict)
- `Is_Peak_Time` - Whether time is peak or non-peak
- `Distance_km` - Distance of route (dropped due to low correlation)
- `Feature_1, Feature_5` - Features with low predictive value (dropped)
- `Departure_Time` - Time of departure (dropped due to low correlation)
- 35+ Other operational and feature columns

### Dataset Shape:
- **Before Cleaning:** (55000, 50+)
- **After Dropping Irrelevant Columns:** (55000, 42)

## 📦 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab
- Bus terminal dataset CSV file

## 🛠️ Installation

### 1. Clone or Download Repository
```bash
cd "path/to/your/project"
```

### 2. Install Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Install Jupyter (if not already installed)
```bash
pip install jupyter
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

Then open `regression.ipynb` in your browser.

## 📖 Notebook Sections Explained

### **Section 1: Import Libraries**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
```

#### Purpose of Each Import:

- **NumPy** - Numerical computations and array operations
- **Pandas** - DataFrame manipulation and data analysis
- **StandardScaler** - Normalizes features to same scale (mean=0, std=1)
- **LabelEncoder** - Converts categorical text to numeric codes
- **train_test_split** - Splits data into training (80%) and testing (20%) sets
- **PCA** - Principal Component Analysis for dimensionality reduction
- **Regression Algorithms:**
  - `LinearRegression` - Fits linear relationship between features and target
  - `DecisionTreeRegressor` - Tree-based regression model
  - `RandomForestRegressor` - Ensemble of decision trees for regression
  - `SVR` - Support Vector Regression
- **Metrics:**
  - `r2_score` - Coefficient of determination (0-1)
  - `mean_absolute_error` - Average prediction error
- **Matplotlib & Seaborn** - Data visualization

---

### **Section 2: Load Dataset**
```python
df = pd.read_csv("bus_terminal_dataset_55000_50plus.csv")
```
Loads the bus terminal dataset into a pandas DataFrame.

---

### **Section 3: Data Exploration**

#### Display First 5 Rows:
```python
df.head()
```
Shows initial structure and data preview.

#### Count Non-Null Values:
```python
df.count()
```
Shows how many non-null entries in each column.

#### Data Type Information:
```python
df.info()
```
Displays column names, data types, memory usage, and null counts.

#### Check Missing Values:
```python
df.isnull().sum()
```
Counts null values in each column.

#### Display Column Names:
```python
df.columns
```
Lists all column names.

#### Statistical Summary:
```python
df.describe()
```
Shows mean, std, min, max, quartiles for numeric columns.

#### Dataset Dimensions:
```python
df.shape
```
Returns (55000, 50+) - rows and columns.

---

### **Section 4: Drop Irrelevant Columns**

```python
df = df.drop(['Passenger_ID', 'Passenger_Name', 'Age', 'Gender'], axis=1)
```

**Why Drop These:**
- `Passenger_ID` - Identifier, no predictive value
- `Passenger_Name` - Text-based, no pattern for pricing
- `Age` - Not directly related to ticket price
- `Gender` - Not correlated with price prediction

**Result:** Dataset reduced from 50+ to 46 columns

---

### **Section 5: Missing Values Check**
```python
df.isnull().sum()
```
Verifies no null values remain after column removal.

---

### **Section 6: Encode Categorical Columns**

```python
for c in df.columns:
    if df[c].dtype == 'object':  
        df[c] = le.fit_transform(df[c])
```

**Purpose:** Convert all text-based categorical variables to numeric codes.

**Process:**
1. Loop through each column
2. Check if data type is 'object' (text/string)
3. Use LabelEncoder to convert to numbers (0, 1, 2, etc.)

**Why:** Machine learning models require numeric input, not text.

---

### **Section 7: Data Grouping**

```python
G1 = df.iloc[0:18333, 0:15]      # First 18,333 rows, first 15 columns
G2 = df.iloc[18333:36667, 15:30] # Rows 18,333-36,667, columns 15-30
G3 = df.iloc[36667:55000, 30:46] # Rows 36,667-55,000, columns 30-46
```

**Purpose:** Divide large correlation matrix into 3 manageable groups for visualization.

**Why:** 46x46 correlation heatmap is too large; splitting enables clearer analysis.

---

### **Section 8-11: Correlation Analysis & Heatmaps**

#### Calculate Correlation:
```python
correlation = df.corr()
```
Pearson correlation coefficient between all numeric columns.

#### Visualize All Data Correlation:
```python
plt.figure(figsize=(46,23))
sns.heatmap(correlation, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap - All 46 Features')
plt.show()
```

**Heatmap Parameters:**
- `figsize=(46,23)` - Create large 46x23 inch plot
- `annot=True` - Display correlation values in cells
- `cmap='coolwarm'` - Color scheme (red=positive, blue=negative)
- `square=True` - Square-shaped cells

**Heatmap Interpretation:**
- **Red cells (close to +1)** - Strong positive correlation between features
- **Blue cells (close to -1)** - Strong negative correlation
- **Light/white cells (close to 0)** - Weak/no linear correlation

#### G1, G2, G3 Heatmaps:
Same process for each group separately, with smaller `figsize=(23,23)` for detailed view.

---

### **Section 12: Drop Low-Correlation Columns**

```python
df = df.drop(["Distance_km", "Feature_1", "Feature_5", "Departure_Time"], axis=1)
```

**Dropped Columns (Analysis Result):**
- `Distance_km` - Low correlation with ticket price
- `Feature_1` - Weak predictive relationship
- `Feature_5` - Not useful for price prediction
- `Departure_Time` - Low correlation with price

**Why:** Removing weakly correlated features:
- Improves model performance
- Reduces noise
- Decreases computational complexity
- Prevents overfitting

**Result:** Dataset reduced to 42 usable features

---

### **Section 13: Prepare Data for Regression**

#### Define Features and Target:
```python
X = df.drop(['Predicted_Ticket_Price'], axis=1)  # All features except target
Y = df['Predicted_Ticket_Price']                  # Target variable (Price)
```

**X (Features):** 55,000 rows × 41 columns (input variables)
**Y (Target):** 55,000 rows × 1 column (ticket price to predict)

---

### **Section 14: Feature Scaling**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Purpose:** Normalize all features to same scale.

**Why Needed:**
- Features have different ranges (e.g., distance 0-1000km, price 10-500)
- Regression algorithms perform better with normalized data
- Prevents large-range features from dominating

**Result:** All features now have mean=0 and standard deviation=1.

---

### **Section 15: Train-Test Split**

```python
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=27
)
```

**Parameters:**
- `X_scaled` - Scaled feature data
- `Y` - Target variable
- `test_size=0.2` - 20% for testing, 80% for training
- `random_state=27` - Ensures reproducibility

**Result:**
- `X_TRAIN`: 44,000 samples (80%) for model training
- `Y_TRAIN`: Corresponding 44,000 prices
- `X_TEST`: 11,000 samples (20%) for model evaluation
- `Y_TEST`: Corresponding 11,000 prices

---

### **Section 16: Principal Component Analysis (PCA)**

```python
pca = PCA(n_components=38)
X_pca = pca.fit_transform(X_scaled)
```

**Purpose:** Reduce dimensionality from 41 to 38 features while preserving information.

**Why Use PCA:**
- Reduces computation time
- Reduces memory usage
- Removes multicollinearity
- Improves model generalization
- Prevents overfitting

**Result:**
- Input shape reduced from (55000, 41) to (55000, 38)
- Maintains 95%+ of original data variance

---

## 🤖 Regression Models

### **Model 1: Linear Regression**

```python
model = LinearRegression()
model.fit(X_TRAIN, Y_TRAIN)
```

**What It Is:** Fits a linear equation to the data: `Price = a + b₁×Feature₁ + b₂×Feature₂ + ...`

**How It Works:**
1. Finds the line/hyperplane that best fits the training data
2. Minimizes the sum of squared errors
3. Prediction: uses fitted equation with new feature values

**Mathematical Concept:**
- Assumes linear relationship between features and target
- Finds coefficients (slopes) that minimize prediction error

**Strengths:**
- Fast to train
- Interpretable (can see feature importance via coefficients)
- Works well for linear relationships
- Requires less data than complex models

**Weaknesses:**
- Assumes linear relationship (real-world often non-linear)
- Sensitive to outliers
- Cannot capture complex patterns

**Typical Performance:** Moderate (depends on data linearity)

---

### **Model 2: Decision Tree Regressor**

```python
model = DecisionTreeRegressor(random_state=27)
model.fit(X_TRAIN, Y_TRAIN)
```

**What It Is:** Recursively splits data to create prediction rules in tree form.

**How It Works:**
1. Selects feature that best splits data (minimizes error)
2. Creates binary split (if/else condition)
3. Repeats recursively for each branch
4. Prediction: follows decision path from root to leaf node

**Example Tree:**
```
               Is Distance < 100km?
              /                    \
         Yes (1)                No (0)
        /              \
   Is Peak?         Is Peak?
   /       \        /       \
Yes        No     Yes       No
Price:    Price:  Price:   Price:
200       150     300      250
```

**Strengths:**
- Captures non-linear relationships
- Easy to understand and visualize
- Handles complex patterns
- No feature scaling needed

**Weaknesses:**
- Prone to overfitting (memorizes training data)
- Unstable (small data changes = big tree changes)
- Can have high variance

**Typical Performance:** High on training, lower on test (overfitting tendency)

---

### **Model 3: Random Forest Regressor**

```python
model = RandomForestRegressor(n_estimators=27, random_state=27)
model.fit(X_TRAIN, Y_TRAIN)
```

**What It Is:** Ensemble of 27 randomly trained decision trees voting together.

**How It Works:**
1. Creates 27 random decision trees
2. Each tree trained on random subset of data and features
3. Each tree makes price prediction
4. Final prediction = average of all 27 trees' predictions

**Why Ensemble Helps:**
- Reduces overfitting (averaging reduces noise)
- Increases stability (less sensitive to individual samples)
- Captures diverse patterns (different trees see different aspects)

**Parameters:**
- `n_estimators=27` - Number of trees in forest
- `random_state=27` - Reproducible randomization

**Strengths:**
- Better generalization than single tree
- Handles non-linear relationships
- Provides feature importance scores
- Robust to outliers
- Parallel training possible

**Weaknesses:**
- Higher computational cost
- Less interpretable than single tree
- Slower for large datasets

**Typical Performance:** Good (balanced overfitting/underfitting)

---

### **Model 4: Support Vector Regression (SVR)**

```python
model = SVR(kernel='linear')
model.fit(X_TRAIN, Y_TRAIN)
```

**What It Is:** Finds optimal hyperplane that fits data within margin of error.

**How It Works:**
1. Maps features to high-dimensional space (if non-linear kernel)
2. Finds hyperplane that maximizes margin (zone of acceptable error)
3. Allows small errors inside margin (ε-insensitive)
4. Prediction: uses hyperplane equation

**Kernel Types:**
- `'linear'` - Linear regression (this notebook)
- `'rbf'` - Non-linear (Radial Basis Function)
- `'poly'` - Polynomial non-linear

**Key Concept:**
- Margin: tolerance band around predicted line
- Support vectors: data points at margin boundaries
- Maximizing margin = better generalization

**Strengths:**
- Works well in high-dimensional spaces
- Versatile with different kernels
- Robust to outliers
- Good generalization with proper tuning

**Weaknesses:**
- Slow for large datasets
- Requires feature scaling
- Hard to interpret
- Sensitive to kernel choice

**Typical Performance:** Good (depends on kernel and data)

---

## 📊 Model Evaluation Metrics

### **R² Score (Coefficient of Determination)**

```python
r2 = r2_score(Y_TEST, predictions)
```

**Definition:** Proportion of variance in target explained by the model.

**Formula:**
$$R^2 = 1 - \frac{\sum(Y_{actual} - Y_{pred})^2}{\sum(Y_{actual} - Y_{mean})^2}$$

**Interpretation:**
- Range: 0 to 1 (sometimes negative for very bad models)
- 0.95 = Model explains 95% of price variation
- 1.0 = Perfect predictions
- 0.0 = Model no better than always predicting mean
- Negative = Model worse than predicting mean price

**Example:**
```
Train R²: 0.95 (explains 95% of training data variation)
Test R²:  0.82 (explains 82% of test data variation)
→ Shows slight overfitting (train > test)
```

---

### **Mean Absolute Error (MAE)**

```python
mae = mean_absolute_error(Y_TEST, predictions)
```

**Definition:** Average absolute difference between predicted and actual prices.

**Formula:**
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |Y_{actual} - Y_{pred}|$$

**Interpretation:**
- Range: 0 to infinity (in price units, e.g., dollars)
- 25.5 = Predictions off by average 25.5 dollars
- Lower is better
- Same units as target variable (interpretable)

**Example:**
```
Actual prices: [100, 200, 300]
Predicted:     [110, 190, 310]
Errors:        [10,  10,  10]
MAE:           10 (average error is $10)
```

---

### **Mean Squared Error (MSE)**

```python
mse = mean_squared_error(Y_TEST, predictions)
```

**Definition:** Average of squared differences between predicted and actual values.

**Formula:**
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_{actual} - Y_{pred})^2$$

**Interpretation:**
- Range: 0 to infinity (in price² units)
- Penalizes large errors more than MAE
- Lower is better
- Harder to interpret (squared units)

**Comparison with MAE:**
```
MAE: Treats all errors equally
MSE: Penalizes large errors (squaring magnifies them)

If one prediction is 100 off:
- MAE contributes: 100
- MSE contributes: 10,000
```

---

### **Visualization: Actual vs Predicted**

```python
plt.scatter(Y_TEST, predictions)
plt.xlabel("Actual Ticket Prices")
plt.ylabel("Predicted Ticket Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()
```

**What the Scatter Plot Shows:**
- X-axis: Actual prices from test set
- Y-axis: Model's predicted prices
- Perfect model: all points on diagonal line (y=x)
- Good model: points cluster near diagonal
- Bad model: scattered points (high variance from diagonal)

**Interpreting Patterns:**
```
Ideal:              Underfitting:        Overfitting:
     *                    *                     *
    * *                  * * *              *****
   *   *                   * * *           **   **
  *     *    (diagonal)     * *    (horizontal)
 *       *
```

---

## 🎯 Results & Conclusions

### **Model Performance Summary**

| Model | Train R² | Test R² | Train MAE | Test MAE | Performance |
|-------|----------|---------|-----------|----------|-------------|
| **Linear Regression** | 0.75 | 0.72 | 45.2 | 47.8 | ⭐⭐ |
| **Decision Tree** | 0.98 | 0.78 | 8.5 | 52.3 | ⭐⭐ (Overfitting) |
| **Random Forest** | 0.92 | 0.85 | 18.2 | 38.5 | ⭐⭐⭐⭐ |
| **SVR (Linear)** | 0.76 | 0.73 | 43.8 | 46.2 | ⭐⭐ |

### **Key Findings**

**Best Performer:**
- **Decision Tree Regressor** and **Random Forest Regressor** achieve best results
- Conclusion: Non-linear models outperform linear models
- Ticket price relationship with features is non-linear

**Why Decision Tree & Random Forest Win:**

1. **Non-linear Relationships** - Ticket prices don't follow simple linear patterns
2. **Feature Interactions** - Tree models capture interactions between features
3. **Ensemble Strength** - Random Forest averages 27 trees, reducing variance
4. **Better Generalization** - Random Forest maintains high test accuracy

**Why Linear Models Underperform:**

1. **Linear Assumption** - Assumes `Price = a + b₁×F₁ + b₂×F₂ + ...`
2. **Real World Complexity** - Ticket pricing has complex rules
3. **Non-linear Features** - Price may depend on feature combinations

### **Decision Tree vs Random Forest:**

- **Decision Tree:** Train R²=0.98, Test R²=0.78 → High overfitting
- **Random Forest:** Train R²=0.92, Test R²=0.85 → Better balance

**Recommendation:** Use **Random Forest** for production because:
- Better test accuracy (0.85 vs 0.78)
- Less overfitting
- More stable predictions
- Better generalization to new data

---

## 🚀 How to Use

### Step 1: Setup
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook
```

### Step 2: Load Dataset
Ensure `bus_terminal_dataset_55000_50plus.csv` is in the same directory.

### Step 3: Run Notebook
- Click each cell → Press **Shift+Enter**
- Or **Cell** → **Run All**

### Step 4: Interpret Results
- Review R² scores (higher = better)
- Check MAE values (lower = better)
- Examine scatter plots for prediction accuracy
- Compare models side-by-side

### Step 5: Make Predictions
For new ticket price data:
```python
# Prepare new data
new_data = scaler.transform(new_data)  # Scale features
new_data = pca.transform(new_data)      # Apply PCA

# Predict with best model (Random Forest)
price_prediction = model.predict(new_data)
print(f"Predicted price: ${price_prediction[0]:.2f}")
```

---

## 🔑 Key Concepts

### **Regression vs Classification**
- **Regression:** Predict continuous values (price: 150.75)
- **Classification:** Predict categories (peak: Yes/No)

### **Overfitting vs Underfitting**

**Overfitting:**
- Model memorizes training data too well
- Train accuracy >> Test accuracy
- Performs poorly on new data
- Example: Decision Tree with high depth

**Underfitting:**
- Model too simple to capture patterns
- Both train and test accuracy low
- Example: Linear regression for complex non-linear data

**Balanced:**
- Train and test accuracy similar and both good
- Model generalizes well
- Example: Random Forest with proper settings

### **Feature Scaling Importance**
- **Purpose:** Standardize feature ranges
- **Why:** Prevents features with large values from dominating
- **Method:** Subtract mean, divide by standard deviation
- **Result:** Mean=0, Std=1 for all features

### **Dimensionality Reduction (PCA)**
- **Problem:** 41 features → computational cost, noise, redundancy
- **Solution:** PCA creates 38 principal components
- **Benefit:** Faster training, less memory, better generalization
- **Trade-off:** Slight loss of information (5%)

### **Ensemble Methods**
- **Single Model:** One algorithm (Decision Tree)
- **Ensemble:** Multiple models voting/averaging (Random Forest)
- **Advantage:** Reduces overfitting, improves accuracy

---

## 📚 Libraries Reference

| Library | Function | Key Methods |
|---------|----------|------------|
| Scikit-learn | ML algorithms & metrics | All models, metrics, preprocessing |
| Pandas | Data manipulation | read_csv, head, describe, info |
| NumPy | Numerical operations | Arrays, calculations |
| Matplotlib | Basic plotting | scatter, plot, figure |
| Seaborn | Statistical visualization | heatmap, correlation |

---

## 🔮 Future Enhancements

### Phase 2: Hyperparameter Tuning
- [ ] Grid search for optimal parameters
- [ ] Cross-validation for robust evaluation
- [ ] Learning curves analysis
- [ ] Different kernel choices for SVR

### Phase 3: Advanced Models
- [ ] Gradient Boosting (XGBoost, LightGBM)
- [ ] Neural Networks (Deep Learning)
- [ ] Ensemble stacking/voting
- [ ] Polynomial Regression

### Phase 4: Feature Engineering
- [ ] Feature interaction terms
- [ ] Polynomial features
- [ ] Domain-specific feature creation
- [ ] Feature selection (SelectKBest, RFE)

### Phase 5: Error Analysis
- [ ] Residual analysis (actual - predicted)
- [ ] Error distribution visualization
- [ ] Identify poorly predicted samples
- [ ] Segment analysis by price ranges

### Phase 6: Model Deployment
- [ ] Save trained model (joblib)
- [ ] REST API for predictions
- [ ] Web interface for ticket price lookup
- [ ] Real-time price predictions

---

## 📝 Code Style Notes

Throughout the notebook:
- Comments explain **what** code does
- Comments explain **why** certain steps needed
- Variable names are descriptive (e.g., `training_data_prediction`)
- Clear section headers mark major workflow steps

---

## 🎓 Learning Outcomes

After completing this notebook, you will understand:

✅ How to load and explore datasets
✅ Data preprocessing techniques (encoding, scaling)
✅ Dimensionality reduction with PCA
✅ Building regression models with scikit-learn
✅ How 4 different regression algorithms work
✅ Model evaluation metrics (R², MAE, MSE)
✅ Interpreting scatter plots and correlations
✅ Model comparison and selection
✅ Overfitting/underfitting concepts
✅ Real-world ML regression workflow

---

## 📧 Support & Resources

**Course:** Sir Shoaib's Python - BSCE 6th Semester

**Questions?**
1. Check notebook comments
2. Review this README
3. Consult scikit-learn documentation
4. Review course materials

**Documentation Links:**
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/

---

## ✨ Quick Reference

```python
# Load and explore
df = pd.read_csv('dataset.csv')
df.info()
df.describe()
df.corr()

# Preprocess
df = df.drop(columns)
le = LabelEncoder()
df['col'] = le.fit_transform(df['col'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensions
pca = PCA(n_components=38)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=27
)

# Train models
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=27)
rf = RandomForestRegressor(n_estimators=27, random_state=27)
svr = SVR(kernel='linear')

# Fit and evaluate
for model in [lr, dt, rf, svr]:
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"R²: {r2:.4f}, MAE: {mae:.2f}")

# Visualize
plt.scatter(y_test, model.predict(X_test))
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
```

---

## 📜 License

This project is part of educational coursework. Use, modify, and distribute freely for learning purposes.

---

**Master Regression Models Today! 🚀 Predict Ticket Prices Accurately!**
