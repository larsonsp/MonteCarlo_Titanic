import subprocess
import pandas as pd
import numpy as np

# Define the expected parameters and the path to the CUDA program
n_values = [100, 10000, 100000,1000000,1500000,2000000,3000000,4000000,5000000,6000000,7000000,8000000]  # Set your desired values of n
categorical_variables = ['Survived', 'Pclass', 'Sex', 'Embarked']
non_categorical_variables = ['Age', 'Fare', 'SibSp', 'Parch']

results = []

for n in n_values:
    # Run the CUDA program with mode 0 and n as arguments
    subprocess.run(["./data", "0", str(n)], check=True)

    # Read the output CSV file
    df = pd.read_csv("output.csv")

    # Calculate mean and standard deviation for non-categorical variables
    variable_stats = {}
    for variable in non_categorical_variables:
        mean = df[variable].mean()
        stddev = df[variable].std()
        variable_stats[variable] = {'mean': mean, 'stddev': stddev}

    # Calculate ratios for categorical variables
    categorical_ratios = {}
    for variable in categorical_variables:
        ratio = df[variable].mean()  # Calculate the ratio instead of mean
        categorical_ratios[variable] = ratio

    results.append({'n': n, 'variable_stats': variable_stats, 'categorical_ratios': categorical_ratios})

# Calculate the initial ratios for categorical variables
initial_categorical_ratios = results[0]['categorical_ratios']

   # Initial means and standard deviations
initial_means = {
    'Age': 29.72063116,
    'Fare': 29.82596385,
    'SibSp': 0.523008,
    'Parch': 0.381594
}

initial_stddevs = {
    'Age': 14.529881995552493,
    'Fare': 38.06890240954017,
    'SibSp': 1.102743,
    'Parch': 0.805605
}

# Print the results
for result in results:
    n = result['n']
    variable_stats = result['variable_stats']
    categorical_ratios = result['categorical_ratios']
    
    print(f"Results for n = {n}:")
    
 

    # Calculate and print mean and standard deviation for non-categorical variables
    for variable in non_categorical_variables:
        initial_mean = initial_means.get(variable, "N/A")
        initial_stddev = initial_stddevs.get(variable, "N/A")
        
        mean = variable_stats[variable]['mean']
        stddev = variable_stats[variable]['stddev']
        
        print(f"{variable}: Initial Mean = {initial_mean}, Initial StdDev = {initial_stddev}, Mean = {mean:.4f}, StdDev = {stddev:.4f}")



    print("Categorical Ratios (compared to initial ratios):")
    for variable in categorical_variables:
        ratio = categorical_ratios[variable]
        initial_ratio = initial_categorical_ratios[variable]
        print(f"{variable}: Ratio = {ratio:.4f} (Initial Ratio = {initial_ratio:.4f})")

    print()
