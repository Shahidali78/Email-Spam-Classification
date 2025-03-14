import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data_path = "C:\AI_Work\My_projects\Email-Spam-Classification\data\emails.csv"
model_path = "models/spam_classifier.pkl"

def load_data():
    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:-1]  
    y = df.iloc[:, -1]  
    return X, y

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully!")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    train_model()
