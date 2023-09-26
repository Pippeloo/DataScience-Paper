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
        model.predict([[263, 62, 0.01, 71.27962362, 65, 1154, 19.1, 83, 6, 8.16, 65, 0.1, 584.25921, 33736494, 17.2, 17.3,0.479, 10.1]])
        # Add the current time measurement to the list
        time_measurements.append(time.time() - current_time)
        # Get the CPU usage
        usage = p.cpu_percent(interval=None)
        # Add the CPU usage measurement to the list
        cpu_usage_measurements.append(usage)
        time.sleep(0.1)

# Call the function to make predictions
predict()

# Save the CPU usage measurements and time measurements to a CSV file
np.savetxt("cpu_usage_measurements.csv", cpu_usage_measurements, delimiter=",")
np.savetxt("time_measurements.csv", time_measurements, delimiter=",")