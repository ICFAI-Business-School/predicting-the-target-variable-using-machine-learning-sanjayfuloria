
### 3. Add the Autograding Test
Create a `tests` directory and add a test script to it. Here's an example `test_regression.py`:

```python
import unittest
import regression

class TestRegression(unittest.TestCase):
    def test_regression(self):
        # Placeholder for actual dataset and model testing
        self.assertTrue(hasattr(regression, 'model'), "The model is not defined in the script")
        self.assertTrue(hasattr(regression, 'mse'), "The mean squared error is not defined in the script")

if __name__ == '__main__':
    unittest.main()
