
#THEN CLEAN AND PRINT AGAIN
#NEED TO MAKE SURE TO FIND THE WRITE ONES FOR LOGNORM

import pandas as pd
import numpy as np
import scipy.stats as st

# Load the dataset
file_path = 'Titanic-Dataset.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

def analyze_dataframe(df):
    # Columns to consider
    columns_to_consider = ['Survived', 'Pclass', 'Age', 'Sex', 'Embarked', 'Fare', 'SibSp']
    # List of distributions to test
    distributions = [st.norm, st.lognorm, st.expon, st.gamma, st.beta, st.weibull_min, st.weibull_max]

    # Print summary statistics for each column
    for column in columns_to_consider:
        print(f"Summary statistics for '{column}':")
        print(df[column].describe(), "\n")

    # Print out distributions
    for variable in columns_to_consider:
        print(f"Distribution for {variable}:")
        if df[variable].dtype == 'object' or variable in ['Survived', 'Pclass']:
            distribution = df[variable].value_counts()
        else:
            distribution = df[variable].describe()
        print(distribution)
        print("\n")

    # Function to fit data to distributions and calculate additional stats
    def fit_distribution(data):
        best_distribution = None
        best_params = None
        best_s = np.inf
        additional_stats = {}

        for distribution in distributions:
            params = distribution.fit(data)
            D, p = st.kstest(data, distribution.name, args=params)
            if D < best_s:
                best_distribution = distribution
                best_params = params
                best_s = D
                if distribution == st.lognorm:
                    s, loc, scale = params
                    mean, var, skew, kurt = st.lognorm.stats(s, loc=loc, scale=scale, moments='mvsk')
                    additional_stats = {'mean': mean, 'stddev': np.sqrt(var)}
        return best_distribution.name, best_params, additional_stats

    # Fit each variable to the best distribution
    best_fits = {}
    for column in df.columns:
        if df[column].dtype != 'O' and column not in ['PassengerId', 'Survived']:
            data = df[column].dropna()
            best_fit_name, best_fit_params, additional_stats = fit_distribution(data)
            best_fits[column] = (best_fit_name, best_fit_params, additional_stats)

    # Print the best fits and additional stats
    for column, (fit_name, fit_params, stats) in best_fits.items():
        print(f"Column: {column}, Best Fit: {fit_name}, Parameters: {fit_params}, Additional Stats: {stats}")

    # Calculating and printing ratios for categorical variables
    categorical_columns = ['Survived', 'Sex', 'Embarked', 'Pclass']
    for column in categorical_columns:
        ratio = df[column].value_counts(normalize=True, dropna=False)
        print(f"\n{column} Ratios:\n", ratio)


def clean_titanic_data(df):
    # Fill missing Age values based on Pclass and Sex
    age_ref = {
        (1, 'female'): 36.0, (1, 'male'): 42.0,
        (2, 'female'): 28.0, (2, 'male'): 29.5,
        (3, 'female'): 22.0, (3, 'male'): 25.0
    }
    for pclass in range(1, 4):
        for sex in ['male', 'female']:
            mask = (df['Age'].isnull()) & (df['Pclass'] == pclass) & (df['Sex'] == sex)
            df.loc[mask, 'Age'] = age_ref[(pclass, sex)]

    # Fill missing Embarked values
    df['Embarked'] = df['Embarked'].fillna('S')

    # Fill missing Fare values
    med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    df['Fare'] = df['Fare'].fillna(med_fare)

    # Convert 'Sex' to numerical
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # Create dummy variables for 'Embarked' without dropping the original column
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)

    # Additional feature engineering can be done here if needed

    return df


#analyze_dataframe(df)
cleaned_df = clean_titanic_data(df)
print("CLEANED STATISTICS")
#analyze_dataframe(cleaned_df)
largeDF = pd.read_csv('output.csv')
analyze_dataframe(largeDF)
