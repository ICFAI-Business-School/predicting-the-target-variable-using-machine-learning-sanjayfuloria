import unittest
import regression

class TestRegression(unittest.TestCase):
    def test_regression(self):
        # Ensure the regression script has the necessary components
        self.assertTrue(hasattr(regression, 'model'), "The model is not defined in the script")
        self.assertTrue(hasattr(regression, 'mse'), "The mean squared error is not defined in the script")

if __name__ == '__main__':
    unittest.main()
