import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class TestRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate a simple dataset
        cls.data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': range(200, 300)
        })

        # Split the dataset into features and target variable
        cls.X = cls.data[['feature1', 'feature2']]
        cls.y = cls.data['target']

        # Split the data into training and testing sets
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

    def test_model_training(self):
        # Initialize and train the regression model
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Predict on the testing set
        y_pred = model.predict(self.X_test)
        
        # Evaluate the model
        mse = mean_squared_error(self.y_test, y_pred)
        
        # Check if the model is trained and predictions are made
        self.assertIsNotNone(y_pred, "Model did not predict any values")
        
        # Check if Mean Squared Error is within an acceptable range
        self.assertLess(mse, 1.0, "Mean Squared Error is too high")
        
        print(f'The Predicted Value: {y_pred}')
        print(f'Mean Squared Error: {mse}')

if __name__ == '__main__':
    unittest.main()
