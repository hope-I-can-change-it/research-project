import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


class ModelAnalyzer:
    def __init__(self, scenarios, epochs):
        self.scenarios = scenarios
        self.epochs = epochs
        self.base_paths = self.create_paths()

    def create_paths(self):
        paths = {}
        for scenario in self.scenarios:
            data_path = os.path.join("trained_models", scenario, f"{self.epochs}_epochs")
            paths[scenario] = data_path
        return paths

    def get_errors(self):
        errors = {}
        for scenario, path in self.base_paths.items():
            errors_per_scenario = []
            for i in range(5):
                version_path = os.path.join(path, str(i))
                config_path = os.path.join(version_path, f"config_{i}.json")

                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)

                    error = config.get("error_u", [])
                    errors_per_scenario.append(float(error))
                except FileNotFoundError:
                    print(f"Warning: File not found at {config_path}")
            errors[scenario] = errors_per_scenario
        return errors

    def get_times(self):
        times = {}
        for scenario, path in self.base_paths.items():
            times_per_scenario = []
            for i in range(5):
                version_path = os.path.join(path, str(i))
                config_path = os.path.join(version_path, f"config_{i}.json")

                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)

                    time = config.get("time_to_train", [])
                    times_per_scenario.append(float(time))
                except FileNotFoundError:
                    print(f"Warning: File not found at {config_path}")
            times[scenario] = times_per_scenario
        return times

    def compute_std_of_error(self):
        errors_all = self.get_errors()
        errors_std = {}
        for scenario, errors in errors_all.items():
            errors_std[scenario] = np.std(errors, ddof=1)
        return errors_std

    def compute_std_of_time(self):
        times_all = self.get_times()
        times_std = {}
        for scenario, times in times_all.items():
            times_std[scenario] = np.std(times)
        return times_std

    def plot_all_errors_averaged(self):
        errors_all = self.get_errors()
        errors_averaged = {}

        for scenario, errors in errors_all.items():
            errors_averaged[scenario] = np.mean(errors)

        # Create the bar chart
        scenario_names = list(errors_averaged.keys())
        errors_averaged_values = list(errors_averaged.values())
        errors_std = list(self.compute_std_of_error().values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(scenario_names, errors_averaged_values, color="skyblue", width=0.5)

        # Format y-axis values in scientific notation
        formatter = FuncFormatter(lambda x, _: f'{x:.2e}')
        plt.gca().yaxis.set_major_formatter(formatter)

        # Add exact values on top of the bars
        for bar, value, std in zip(bars, errors_averaged_values, errors_std):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{value:.2e} ± {std:.2e}',
                ha='center',
                va='bottom',
                fontsize=10,
                color='black'
            )

        plt.xlabel("Model Name")
        plt.ylabel("Average L2 Error")
        plt.title(f"Comparison of Average L2 Error Across Scenarios ({self.epochs} epochs)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.show()

    def plot_all_times_averaged(self):
        times_all = self.get_times()
        times_averaged = {}

        for scenario, times in times_all.items():
            times_averaged[scenario] = np.mean(times)

        # Create the bar chart
        scenario_names = list(times_averaged.keys())
        times_averaged_values = list(times_averaged.values())
        times_std = list(self.compute_std_of_time().values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(scenario_names, times_averaged_values, color="blue", width=0.5)

        # Add exact values on top of the bars
        for bar, value, std in zip(bars, times_averaged_values, times_std):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{value:.2f} ± {std:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                color='black'
            )

        plt.xlabel("Model Name")
        plt.ylabel("Average Time in Seconds")
        plt.title(f"Comparison of Average Time Across Scenarios ({self.epochs} epochs)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.show()
