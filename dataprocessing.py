import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from memory_profiler import profile
import plotly.graph_objects as go

# Load the dataset
data_path = 'C:/Users/user/Desktop/col/Plant-Health-Monitoring/test_data.csv'
df = pd.read_csv(data_path)

# Drop non-numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Data Visualization with Plotly

# Correlation heatmap using Plotly
correlation_matrix = df_numeric.corr()
fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                 x=correlation_matrix.columns,
                                 y=correlation_matrix.columns))
fig.update_layout(title='Correlation Heatmap')
fig.show()

# Data processing and outlier detection...

# Define file paths
project_dir = 'C:/Users/user/Desktop/col/Plant-Health-Monitoring'
input_data_file = 'test_data.csv'
output_file = 'preprocessed_dataset.csv'

# Load the dataset
data_path = f'{project_dir}/{input_data_file}'
df = pd.read_csv(data_path)

# Identify missing values
missing_values = df.isnull().sum()

# Convert non-numeric values to NaN
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Drop non-numeric columns
df_numeric = df_numeric.dropna(axis=1)

# Compute mean and standard deviation
mean_values = df_numeric.mean()
std_values = df_numeric.std()

# Calculate Z-scores
z_scores = np.abs((df_numeric - mean_values) / std_values)

# Define threshold for Z-score
threshold = 3

# Identify outliers using Z-score
outliers_z_score = df_numeric[(z_scores > threshold).any(axis=1)]

# Instantiate the Isolation Forest model
model = IsolationForest()

# Fit the model to the data
model.fit(df_numeric)

# Predict outliers
outliers = model.predict(df_numeric)

# Find indices of outliers (if -1, it's an outlier)
outlier_indices = df_numeric.index[outliers == -1]

# Print the indices of outliers detected by Isolation Forest
print("Indices of outliers detected by Isolation Forest:")
print(outlier_indices)

# Data Visualization

# Pair plot
sns.pairplot(df_numeric)
plt.title('Pairwise Relationships Between Numerical Variables')
plt.show()

# Correlation heatmap using seaborn
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Histograms of numerical features
df_numeric.hist(figsize=(12, 8))
plt.suptitle('Histograms of Numerical Features', y=0.95)
plt.show()

# Visualize distribution of numeric features to detect anomalies (e.g., outliers)
plt.figure(figsize=(10, 6))
df_numeric.boxplot()
plt.title('Boxplot of Numeric Features')
plt.xticks(rotation=45)
plt.show()

# Compute correlation matrix
correlation_matrix = df_numeric.corr()

# Create heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df_numeric.describe())
