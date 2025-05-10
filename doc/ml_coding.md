
# Machine Learning Problems

## Problem 1: Data Pipeline Optimization
### Task:
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
### Task:
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
### Task:
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
### Task:
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
### Task:
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
