import subprocess

# Define parameter ranges
epochs_list = [10000]  # List of different epoch values to test
scenarios = [5]  # List of scenarios
num_repeats = [0, 1, 2, 3, 4]  # List of versions

# Loop over all combinations
for epochs in epochs_list:
    for scenario in scenarios:
        for version in num_repeats:
            print(f"Starting training: epochs={epochs}, scenario={scenario}, version={version}")

            # Run the training script with different parameters
            command = [
                "python", "burgers.py",
                "--epochs", str(epochs),
                "--scenario", str(scenario),
                "--version", str(version)
            ]

            subprocess.run(command)

print("All training runs completed.")
