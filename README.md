[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/g_0BYp0d)
# Machine_Learning_Assignment
This is an assignment to the MA Economics Machine Learning Class

# Basic Machine Learning Task: Regression

## Objective
The objective of this assignment is to write a simple Python program to perform a basic machine learning task using regression.

## Instructions
1. Fork this repository to your GitHub account.
2. Clone the forked repository to your local machine.
3. Write a Python script named `regression.py` that:
   - Loads a dataset (you can use any dataset of your choice).
   - Splits the dataset into training and testing sets.
   - Trains a regression model (e.g., Linear Regression) on the training set.
   - Evaluates the model on the testing set and prints the evaluation metrics (e.g., Mean Squared Error).
4. Commit your changes and push them to your forked repository.

## Requirements
- Use Python 3.x.
- Use libraries such as `pandas`, `numpy`, `scikit-learn`, etc.

## Submission
Submit the URL of your forked repository with the completed assignment.

## Example
Here is a basic structure of the `regression.py` script:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv('path/to/your/dataset.csv')

# Split the dataset into features and target variable
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

