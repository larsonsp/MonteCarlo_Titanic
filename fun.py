import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

def preprocess_data(df):
    # Preprocess the data
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    return df

def train_logistic_regression(df, target_variable):
    # Split the data
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Evaluate Logistic Regression
    y_pred = log_reg.predict(X_test)
    print("Logistic Regression Evaluation Report:")
    print(classification_report(y_test, y_pred))
    
    return log_reg

def predict_survival(log_reg_model, passenger_data):
    # Ensure the passenger data is scaled before prediction
    scaler = StandardScaler()
    passenger_data_scaled = scaler.fit_transform([passenger_data])
    prediction = log_reg_model.predict(passenger_data_scaled)
    survived = "Yes" if prediction[0] == 1 else "No"
    return survived


if __name__ == "__main__":
    # Load and preprocess your dataset (replace 'dataset.csv' with your actual dataset path)
    dataset = preprocess_data(pd.read_csv('Titanic-Dataset.csv'))

    # Train Logistic Regression and select features
    target_variable = 'Survived'
    log_reg_model = train_logistic_regression(dataset, target_variable)
    

    while True:
        try:
            # Input passenger information with all features used in the model
            pclass = int(input("Enter Pclass (1/2/3): "))
            sex = int(input("Enter Sex (0 for male, 1 for female): "))
            age = float(input("Enter Age: "))  # Assuming Age is a feature
            sibsp = int(input("Enter number of siblings/spouses aboard: "))  # Assuming SibSp is a feature
            parch = int(input("Enter number of parents/children aboard: "))  # Assuming Parch is a feature
            fare = float(input("Enter Fare: "))  # Assuming Fare is a feature
            embarked = int(input("Enter Embarked (0 for C, 1 for Q, 2 for S): "))
            
            passenger_data = [pclass, sex, age, sibsp, parch, fare, embarked]

            # Predict survival
            result = predict_survival(log_reg_model, passenger_data)
            print(f"Based on the Logistic Regression model, the passenger would have survived: {result}")

        except ValueError:
            print("Invalid input. Please enter valid numeric values.")
        
        choice = input("Do you want to make another prediction? (yes/no): ")
        if choice.lower() != 'yes':
            break
