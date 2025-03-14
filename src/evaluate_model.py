import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# File paths
data_path = "C:\AI_Work\My_projects\Email-Spam-Classification\data\emails.csv"
model_path = "models/spam_classifier.pkl"

def load_data():
    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:-1]  
    y = df.iloc[:, -1]  
    return X, y

def evaluate_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
