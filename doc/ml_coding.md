
# Machine Learning Problems

## Question 1: Bias-Variance Tradeoff (Easy)  
What is the bias-variance tradeoff in machine learning, and how does it affect model performance?  

**Answer:**
- **Bias:** The error introduced when a model is too simple and fails to capture the underlying pattern. High bias can lead to underfitting.  
- **Variance:** The error introduced when a model is too complex and captures noise rather than the underlying pattern. High variance can lead to overfitting.  
- **Tradeoff:**  
  - Increasing model complexity reduces bias but increases variance.  
  - Reducing model complexity decreases variance but increases bias.  
- **Solution:**  
  - Use cross-validation to find a balance between bias and variance.  
  - Regularization techniques (like L2 regularization) can help reduce variance without significantly increasing bias.  

---

## Question 2: Overfitting vs. Underfitting (Easy)  
What are overfitting and underfitting, and how can you identify and mitigate them?  

**Answer:**  
- **Overfitting:**  
  - The model learns the noise in the training data, leading to poor generalization on new data.  
  - **Signs:** High accuracy on training data but low accuracy on validation data.  
  - **Mitigation:** Regularization (L1, L2), cross-validation, simpler models, data augmentation.  

- **Underfitting:**  
  - The model is too simple to capture the underlying trend, resulting in poor performance on both training and validation data.  
  - **Signs:** Low accuracy on both training and test data.  
  - **Mitigation:** Increase model complexity, add more features, reduce regularization.  

---

## Question 3: Cross-Validation (Easy)  
Why is cross-validation important in machine learning, and how does k-fold cross-validation work?  

**Answer:**  
- **Importance:**  
  - Provides a more reliable estimate of model performance compared to a single train-test split.  
  - Helps in detecting overfitting and underfitting.  
- **K-Fold Cross-Validation:**  
  - The data is split into k equal parts (folds).  
  - The model is trained on k-1 folds and tested on the remaining fold.  
  - This process repeats k times, with each fold used as the test set once.  
  - The average performance across all k folds gives a more accurate estimate.  

---

## Question 4: Regularization (Easy)  
What is regularization, and why is it important in machine learning?  

**Answer:**  
- **Definition:** Regularization adds a penalty to the loss function to prevent overfitting by constraining model complexity.  
- **Types:**  
  - **L1 Regularization (Lasso):** Adds the absolute value of the coefficients as a penalty. Can perform feature selection.  
  - **L2 Regularization (Ridge):** Adds the squared value of the coefficients as a penalty. Helps in preventing large weights.  
- **Importance:**  
  - Reduces model variance without significantly increasing bias.  
  - Helps maintain generalization by avoiding overly complex models.  

---

## Question 5: Gradient Descent (Easy)  
What is gradient descent, and how does it work?  

**Answer:**  
- **Definition:** An optimization algorithm used to minimize the loss function by iteratively adjusting the model parameters.  
- **Steps:**  
  1. Initialize the parameters randomly.  
  2. Calculate the gradient of the loss function.  
  3. Update the parameters in the direction opposite to the gradient.  
  4. Repeat until convergence.  
- **Learning Rate:**  
  - Controls the size of each step.  
  - A high learning rate may overshoot the minimum, while a low rate may take too long to converge.  
- **Variants:**  
  - **Batch Gradient Descent:** Uses the entire dataset.  
  - **Stochastic Gradient Descent (SGD):** Uses one data point at a time.  
  - **Mini-Batch Gradient Descent:** Uses a small batch of data points.  

---

## Question 6: Dimensionality Reduction (Easy)  
What is dimensionality reduction, and why is it used in machine learning?  

**Answer:**  
- **Definition:** The process of reducing the number of input variables while retaining the most important information.  
- **Techniques:**  
  - **Principal Component Analysis (PCA):** Projects data into a lower-dimensional space using eigenvectors.  
  - **t-SNE:** Non-linear technique for visualizing high-dimensional data.  
- **Why Use It:**  
  - Reduces computational cost and memory usage.  
  - Mitigates the curse of dimensionality.  
  - Helps visualize data in 2D or 3D.  

---

## Question 7: Ensemble Learning (Easy)  
What is ensemble learning, and why is it effective?  

**Answer:**  
- **Definition:** Combines the predictions of multiple models to improve accuracy.  
- **Types:**  
  - **Bagging:** Trains multiple models independently and averages their predictions (e.g., Random Forest).  
  - **Boosting:** Trains models sequentially, where each model focuses on correcting the errors of the previous one (e.g., AdaBoost, Gradient Boosting).  
- **Effectiveness:**  
  - Reduces variance (bagging), bias (boosting), or both.  
  - More robust to overfitting compared to a single model.  

---

## Question 8: Precision vs. Recall (Easy)  
What is the difference between precision and recall, and when would you prioritize one over the other?  

**Answer:**  
- **Precision:** The proportion of true positives among all positive predictions.  
- **Recall:** The proportion of true positives among all actual positives.  
- **When to Prioritize:**  
  - **Precision:** When false positives are costly (e.g., spam detection).  
  - **Recall:** When false negatives are costly (e.g., cancer diagnosis).  
- **Trade-off:** Increasing precision often decreases recall and vice versa. The F1 score is a harmonic mean of precision and recall to balance both.  


## Problem 1: Data Pipeline Optimization
You are given a dataset `system_logs.csv` containing columns: `timestamp`, `cpu_util`, `mem_usage`, `disk_io`, `temp`. 
- Your task is to implement a data pipeline that:
  - Loads the data
  - Cleans missing values by interpolation
  - Aggregates data by hour

### Code Skeleton:
```python
import pandas as pd

def load_data(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    # Interpolate missing values in numeric columns
    # TODO: Implement interpolation
    return df

def aggregate_by_hour(df):
    # Aggregate data by hour using the timestamp column
    # TODO: Group by hour and calculate mean
    return df

# Load data
data = load_data('system_logs.csv')

# Clean data
cleaned_data = clean_data(data)

# Aggregate by hour
hourly_data = aggregate_by_hour(cleaned_data)

print(hourly_data.head())
```
**Instructions:**
1. Implement the `clean_data` function to fill missing values using linear interpolation.
2. Complete the `aggregate_by_hour` function to group data by hour and compute the mean.
3. Test the functions using a sample data file.
---

## Problem 2: Predictive Maintenance with Logistic Regression
Given the dataset `hardware_failures.csv`, predict hardware failure using logistic regression. 
- The dataset has columns: `cpu_util`, `mem_usage`, `disk_io`, `temp`, `failure`.

### Code Skeleton:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_and_prepare_data(file_path):
    # TODO: Load the CSV file and split into features and labels
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # TODO: Train a logistic regression model
    model = LogisticRegression()
    return model

def evaluate_model(model, X_test, y_test):
    # TODO: Predict and calculate accuracy
    return accuracy

# Load and prepare data
X_train, X_test, y_train, y_test = load_and_prepare_data('hardware_failures.csv')

# Train model
model = train_model(X_train, y_train)

# Evaluate model
accuracy = evaluate_model(model, X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
```
**Instructions:**
1. Implement the data loading and preparation function.
2. Train a logistic regression model.
3. Print the accuracy of the model.
---

## Problem 3: Real-Time Anomaly Detection Plotting
You have a streaming dataset `sensor_data.csv` containing columns: `timestamp`, `cpu_util`, `anomaly_flag`. 
- Plot the CPU utilization over time, highlighting anomalies.

### Code Skeleton:
```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_anomalies(df):
    plt.figure(figsize=(10, 5))
    # TODO: Plot CPU utilization over time, marking anomalies
    plt.xlabel('Timestamp')
    plt.ylabel('CPU Utilization')
    plt.title('Real-Time Anomaly Detection')
    plt.show()

# Load data
df = pd.read_csv('sensor_data.csv')
plot_anomalies(df)
```
**Instructions:**
1. Implement the `plot_anomalies` function to plot CPU utilization.
2. Use a different color or marker for anomaly points.
---

## Problem 4: Model Performance Visualization
Train a decision tree classifier on the dataset `performance_data.csv`, containing features: `feature1`, `feature2`, `feature3`, `target`. 
- Plot the ROC curve and calculate the AUC.

### Code Skeleton:
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def train_decision_tree(X, y):
    # TODO: Train a decision tree classifier
    return model

def plot_roc_curve(y_true, y_pred):
    # TODO: Plot ROC curve and calculate AUC
    plt.show()

# Load and prepare data
X, y = load_data('performance_data.csv')

# Train model
model = train_decision_tree(X, y)

# Predict probabilities
y_pred = model.predict_proba(X)[:, 1]

# Plot ROC curve
plot_roc_curve(y, y_pred)
```
**Instructions:**
1. Implement the decision tree training function.
2. Complete the function to plot the ROC curve.
3. Print the AUC value.
---

## Problem 5: Clustering for Anomaly Detection
Cluster the data from `metrics.csv` using K-Means to identify anomalous data points. 
- Columns: `cpu_util`, `mem_usage`, `disk_io`, `temp`.

### Code Skeleton:
```python
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

def cluster_data(df, n_clusters=3):
    # TODO: Apply K-Means clustering and return cluster labels
    return df

def plot_clusters(df):
    plt.figure(figsize=(8, 6))
    # TODO: Plot clusters in 2D space (e.g., cpu_util vs. mem_usage)
    plt.xlabel('CPU Utilization')
    plt.ylabel('Memory Usage')
    plt.title('Clustered Data')
    plt.show()

# Load data
df = pd.read_csv('metrics.csv')

# Cluster data
clustered_df = cluster_data(df)

# Plot clusters
plot_clusters(clustered_df)
```
**Instructions:**
1. Implement the K-Means clustering function.
2. Plot the clusters based on `cpu_util` and `mem_usage`.
3. Highlight the most anomalous cluster.
