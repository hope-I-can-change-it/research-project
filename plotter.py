from model_analyzer import ModelAnalyzer


def run_plotter():
    scenarios, epochs = ["all_off", "all_on", "boundary_on", "col_on", "initial_on", "all_but_col"], 10000
    analyzer = ModelAnalyzer(scenarios, epochs)
    analyzer.plot_all_errors_averaged()
    analyzer.plot_all_times_averaged()

    std_error = analyzer.compute_std_of_error()
    std_time = analyzer.compute_std_of_time()

    formatted_std_error = {key: f'{value:.2e}' for key, value in std_error.items()}
    formatted_std_time = {key: f'{value:.2f}' for key, value in std_time.items()}

    print("Standard Deviation of Error:")
    print(formatted_std_error)

    print("Standard Deviation of Time (rounded to 2 decimal places):")
    print(formatted_std_time)


run_plotter()
