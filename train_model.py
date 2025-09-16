import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load your dataset (update path if needed)
DATA_PATH = r"C:\Users\Pooja\Desktop\Student Performance\student_info.csv"
df = pd.read_csv(DATA_PATH)

# Select 5 features and target
X = df[['age', 'health', 'absences', 'G1', 'G2']]
y = df['G3']  # final grade

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
MODEL_PATH = os.path.join("C:\\Users\\Pooja\\Desktop\\Student Performance", "student_model_lr.pkl")
joblib.dump(model, MODEL_PATH)

print("Model trained and saved successfully!")
