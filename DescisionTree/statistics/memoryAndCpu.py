import joblib  # Import the joblib library
import numpy as np  # If you need NumPy for data manipulation
from memory_profiler import profile
import time 
import psutil
import matplotlib.pyplot as plt

cpu_usage_measurements = []
time_measurements = []

# Load the saved model
model = joblib.load("model.pkl")

p = psutil.Process(pid=psutil.Process().pid)
p.cpu_percent(interval=None)

# # Use the loaded model to make predictions
@profile()
def predict():
    # Set the current time
    current_time = time.time()
    # Loop the specified number of times
    for i in range(10):
        # Make a prediction for the specified features.
        model.predict([[5, 1.5], [3, 2]])
        # Add the current time measurement to the list
        time_measurements.append(time.time() - current_time)
        # Get the CPU usage
        usage = p.cpu_percent(interval=None)
        # Add the CPU usage measurement to the list
        cpu_usage_measurements.append(usage)
        # time.sleep(0.01)

# Call the function to make predictions
predict()

# Save the CPU usage measurements and time measurements to a CSV file
np.savetxt("cpu_usage_measurements.csv", cpu_usage_measurements, delimiter=",")
np.savetxt("time_measurements.csv", time_measurements, delimiter=",")