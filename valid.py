import subprocess
import pandas as pd
import numpy as np


# Define the start, end, and the number of values
start = 10000
end = 1000000
num_values = 20  # You can change this to get more or fewer values

n_values = np.linspace(start, end, num_values, dtype=int)




# Define the expected parameters and the path to the CUDA program
#n_values = [100, 10000, 100000, 1000000, 1500000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000]
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

# Initial values
initial_means = {'Age': 29.72063116, 'Fare': 29.82596385, 'SibSp': 0.523008, 'Parch': 0.381594}
initial_stddevs = {'Age': 14.529881995552493, 'Fare': 38.06890240954017, 'SibSp': 1.102743, 'Parch': 0.805605}
initial_categorical_ratios = results[0]['categorical_ratios']

# Prepare vectors for R data frame
n_vector = "c(" + ", ".join(str(result['n']) for result in results) + ")"
mean_vectors = {var: "c(" + ", ".join(f"{result['variable_stats'][var]['mean']:.4f}" for result in results) + ")" for var in non_categorical_variables}
stddev_vectors = {var: "c(" + ", ".join(f"{result['variable_stats'][var]['stddev']:.4f}" for result in results) + ")" for var in non_categorical_variables}
ratio_vectors = {var: "c(" + ", ".join(f"{result['categorical_ratios'][var]:.4f}" for result in results) + ")" for var in categorical_variables}

# Print vectors for R
print("n <- " + n_vector)
for var in non_categorical_variables:
    print(f"{var}_mean <- " + mean_vectors[var])
    print(f"{var}_stddev <- " + stddev_vectors[var])
for var in categorical_variables:
    print(f"{var}_ratio <- " + ratio_vectors[var])

# Print initial values for reference
print("\n# Initial Values for Reference")
print("initial_means <-", initial_means)
print("initial_stddevs <-", initial_stddevs)
print("initial_categorical_ratios <-", initial_categorical_ratios)
