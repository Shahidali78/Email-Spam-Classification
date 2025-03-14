import pickle
import numpy as np
import pandas as pd

data_path = "models/spam_classifier.pkl"

# Load the trained model
with open(data_path, 'rb') as file:
    model = pickle.load(file)

# Extract feature names from the model
try:
    feature_names = model.feature_names_in_ 
except AttributeError:
    print("Error: Model does not contain feature names. Retrain it with updated scikit-learn version.")

def predict_email(email_vector):
    email_df = pd.DataFrame([email_vector], columns=feature_names)
    
    # Make the prediction
    prediction = model.predict(email_df)
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    example_email_vector = np.random.randint(0, 10, size=len(feature_names))
    
    print("Prediction:", predict_email(example_email_vector))
