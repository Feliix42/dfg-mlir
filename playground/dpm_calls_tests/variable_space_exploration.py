import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

# Executable and parameter settings
executable = "./build/lms_sample.elf"  # Replace with the actual path to your executable
mappings = ["mapping1", "mapping2", "mapping3", "mapping4"]
token_counts = range(1000, 20001, 1000)
num_trials = 3  # Number of trials for each configuration

# Dictionary to store the average results for each mapping
results = {mapping: [] for mapping in mappings}

# Run tests with averaging
for mapping in mappings:
    for token_count in token_counts:
        # Store execution times for multiple trials
        trial_times = []

        for _ in range(num_trials):
            # Set the environment variable
            env = {"DFG_TOKEN_COUNT": str(token_count)}

            # Start the timer, execute the process, and measure the time
            start_time = time.time()
            process = subprocess.run([executable,  "-c",  "yaml_config/LMS_intel_i5.yaml", "-m", mapping], env=env, capture_output=True)
            end_time = time.time()

            # Calculate elapsed time and add to trial times
            elapsed_time = end_time - start_time
            trial_times.append(elapsed_time)
            print(elapsed_time)

        # Calculate the average time for this configuration
        avg_time = np.min(trial_times)
        results[mapping].append(avg_time)

        # Print each step for monitoring
        print(f"Mapping: {mapping}, DFG_TOKEN_COUNT: {token_count}, Average Time: {avg_time:.2f} seconds")

# Plot the results
plt.figure(figsize=(10, 6))
for mapping in mappings:
    plt.plot(token_counts, results[mapping], label=mapping)

plt.xlabel("DFG_TOKEN_COUNT")
plt.ylabel("Average Execution Time (seconds)")
plt.title("Average LMS Signature Execution Time by Mapping and Token Count")
plt.legend()
plt.grid(True)
plt.show()

