import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Load your dataset
file_path = "/content/student_data (1).csv"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    print("Please upload the 'student_data (1).csv' file to your Colab environment.")
else:
    df = pd.read_csv(file_path)

    # Print column names to help diagnose KeyError
    print("DataFrame columns:", df.columns.tolist())

    # Target column name
    target_col = "G3"  # make sure it matches your CSV exactly

    # Encode all categorical (object/string) columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))

    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=5000) # Increased max_iter
    model.fit(X_train, y_train)

    # Evaluate on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

    # Save the trained model to a file
    joblib.dump(model, "student_model_lr.pkl")
    print("💾 Model saved as student_model_lr.pkl")