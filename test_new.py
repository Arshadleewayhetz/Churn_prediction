import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from app.models.model import preprocess_data

def load_model(model_path):
    return joblib.load(model_path)

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    X, _ = preprocess_data(df)
    return X

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def main():
    model_path = 'path/to/your/trained_model.pkl'
    test_data_path = 'path/to/your/new_test_data.csv'

    model = load_model(model_path)
    X_test = load_and_preprocess_data(test_data_path)
    y_test = X_test.pop('Churn')  # Assuming 'Churn' is the target variable

    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy on new test dataset: {accuracy:.2f}")

if __name__ == "__main__":
    main()