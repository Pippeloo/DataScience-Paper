import joblib  # Import the joblib library
import numpy as np  # If you need NumPy for data manipulation
from memory_profiler import profile
import time

# Load the saved model
model = joblib.load("model.pkl")

# Use the loaded model to make predictions
@profile()
def predict():
    # Loop the specified number of times
    for i in range(1):
        # Make a prediction for the specified features.
        model.predict([[263, 62, 0.01, 71.27962362, 65, 1154, 19.1, 83, 6, 8.16, 65, 0.1, 584.25921, 33736494, 17.2, 17.3,0.479, 10.1]])

# Call the function to make predictions
predict()