
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import subprocess


def run_original_data(arg1, model_choice):
    # Load the dataset
    data = pd.read_csv('Titanic-Dataset.csv')

    # Preprocess the data
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Select features based on model choice
    if model_choice in ["1", "2", "3"]:
        X = data.drop('Survived', axis=1)
    elif model_choice in ["4", "5", "6"]:
        X = data[['Pclass', 'Sex', 'Fare']]
    else:
        print("Invalid choice. Please choose a valid model option.")
        return

    # Split the data
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    if model_choice in ["1", "4"]:
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        print("Logistic Regression")
        print(classification_report(y_test, log_reg.predict(X_test)))

    # Random Forest
    elif model_choice in ["2", "5"]:
        rf_clf = RandomForestClassifier(n_estimators=100)
        rf_clf.fit(X_train, y_train)
        print("Random Forest")
        print(classification_report(y_test, rf_clf.predict(X_test)))

    # TensorFlow Neural Network
    elif model_choice in ["3", "6"]:
        model = Sequential([
            Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        print("Neural Network w/ TensorFlow")
        model.evaluate(X_test, y_test)

    else:
        print("Invalid choice. Please choose a valid model option.")




def generate_synthetic_data(n, model_choice):
    # Run the CUDA program with mode 0 and n as arguments
    subprocess.run(["./data", "0", str(n)], check=True)
    data = pd.read_csv('output.csv')

    # Select features based on model choice
    if model_choice in ["1", "2", "3"]:
        X = data.drop('Survived', axis=1)
    elif model_choice in ["4", "5", "6"]:
        X = data[['Pclass', 'Sex', 'Fare']]
    else:
        print("Invalid choice. Please choose a valid model option.")
        return

    # Split the data
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    if model_choice in ["1", "4"]:
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        print("Logistic Regression")
        print(classification_report(y_test, log_reg.predict(X_test)))

    # Random Forest
    elif model_choice in ["2", "5"]:
        rf_clf = RandomForestClassifier(n_estimators=100)
        rf_clf.fit(X_train, y_train)
        print("Random Forest")
        print(classification_report(y_test, rf_clf.predict(X_test)))

    # TensorFlow Neural Network
    elif model_choice in ["3", "6"]:
        model = Sequential([
            Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        print("Neural Network w/ TensorFlow")
        model.evaluate(X_test, y_test)

    else:
        print("Invalid choice. Please choose a valid model option.")


def main():
    while True:
        print("Choose an option:")
        print("1. Run original data")
        print("2. Generate synthetic data")
        print("3. Exit")
        data_choice = input("Enter your choice (1/2/3): ")

        if data_choice == '1':
            arg1 = input("Press zero to Start: ")
            print("Choose a model:")
            print("1. Logistic Regression")
            print("2. Random Forest")
            print("3. TensorFlow Neural Network")
            print("4. Logistic Regression with variable selction")
            print("5. Random Forest with variable selction")
            print("6. TensorFlow Neural Network with variable selction")
            model_choice = input("Enter your choice (1/2/3/4/5/6): ")
            run_original_data(arg1, model_choice)
        elif data_choice == '2':
            n = input("Enter the size of synthetic data (n): ")
            print("Choose a model:")
            print("1. Logistic Regression")
            print("2. Random Forest")
            print("3. TensorFlow Neural Network")
            print("4. Logistic Regression with variable selction")
            print("5. Random Forest with variable selction")
            print("6. TensorFlow Neural Network with variable selction")
            model_choice = input("Enter your choice (1/2/3/4/5/6): ")
            generate_synthetic_data(n, model_choice)
        elif data_choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
